import os
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import faiss
import difflib
from PIL import Image
from torch.distributions import Categorical
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration,
    CLIPVisionModel,
    CLIPImageProcessor
)
from PIL import Image

# ====== lmdeploy 相关导入 ======
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

# ==========================================
# 1. 全局配置
# ==========================================
CONFIG = {
    # --- 路径配置 ---
    "data_path": "./traindata.json",       # 你的训练集/知识库路径
    "test_path": "./testdata.json",        # 你的测试集路径

    
    # --- 模型路径 (请替换为你本地的绝对路径) ---
    "qwen_model_path": "/home/LLMs/Qwen2.5-VL-7B-Instruct", 
    "bge_model_path": "/home/BAAI/bge-large-zh-v1.5",            
    "clip_model_path": "/home/openai/clip-vit-base-patch32",
    "bert_model_path": "/home/google-bert/bert-base-chinese",  # BERTScore 模型路径
    
    # --- 硬件配置 ---
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # --- RL超参数 (REINFORCE) ---
    "embedding_dim": 3584, # BGE + CLIP
    "hidden_dim": 512,
    "top_k_candidates": 50,       # 初筛数量
    "k_shots": 1,                 # 最终给大模型看几个例子
    "learning_rate": 1e-5,        # 策略网络学习率
    "epochs": 8                # 训练轮数
}

# ==========================================
# 2. 多模态编码器 (保持不变)
# ==========================================
class MultimodalEncoder:
    def __init__(self):
        print("正在加载编码模型 (BGE + CLIP)...")
        self.device = CONFIG["device"]
        
        # 加载 BGE (文本)
        self.bge_tokenizer = AutoTokenizer.from_pretrained(CONFIG["bge_model_path"])
        self.bge_model = AutoModel.from_pretrained(CONFIG["bge_model_path"]).to(self.device)
        self.bge_model.eval()
        
        # 加载 CLIP (图像)
        self.clip_processor = CLIPImageProcessor.from_pretrained(CONFIG["clip_model_path"])
        self.clip_model = CLIPVisionModel.from_pretrained(CONFIG["clip_model_path"]).to(self.device)
        self.clip_model.eval()
        
    def get_embedding(self, text, img_path):
        """生成拼接后的 [Text_Emb; Image_Emb]"""
        with torch.no_grad():
            inputs = self.bge_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            text_out = self.bge_model(**inputs)
            text_emb = torch.nn.functional.normalize(text_out.last_hidden_state[:, 0, :], p=2, dim=1)

        full_img_path =  img_path
        try:
            image = Image.open(full_img_path).convert("RGB")
            with torch.no_grad():
                inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
                image_emb = torch.nn.functional.normalize(self.clip_model(**inputs).pooler_output, p=2, dim=1)
        except Exception as e:
            print(f"无法读取图片 {full_img_path}: {e}, 使用全零向量代替")
            image_emb = torch.zeros_like(text_emb)

        combined_emb = torch.cat([text_emb, image_emb], dim=1) 
        return combined_emb.cpu().numpy()

# ==========================================
# 3. 数据加载与索引构建 (保持不变)
# ==========================================
class RAGDataset:
    def __init__(self, json_path, encoder: MultimodalEncoder, is_training=True):
        print(f"正在加载数据: {json_path} ...")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.encoder = encoder
        self.embeddings = []
        
        cache_file = json_path + ".npy"
        if os.path.exists(cache_file):
            print(f"发现缓存特征文件 {cache_file}，直接加载...")
            self.embeddings = np.load(cache_file)
        else:
            print("开始计算多模态特征 (这可能需要一些时间)...")
            emb_list = []
            for item in self.data:
                emb = self.encoder.get_embedding(item['text'], item['image_path'])
                emb_list.append(emb)
            self.embeddings = np.vstack(emb_list).astype('float32')
            np.save(cache_file, self.embeddings)
            
        self.index = None
        if is_training:
            print("构建 Faiss 索引...")
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            self.index.add(self.embeddings)

    def search_candidates(self, query_emb):
        if self.index is None: raise ValueError("Index not initialized")
        D, I = self.index.search(query_emb.reshape(1, -1), CONFIG["top_k_candidates"])
        return I[0], self.embeddings[I[0]]

    def get_item(self, idx):
        return self.data[idx]

# ==========================================
# 4. 策略网络 (Policy Network) - REINFORCE
# ==========================================
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        # 确保这里的 input_dim 是拼接后的维度（2048 或 3584）
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # input_dim 修改为 3584 或实际拼接后的维度
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # 输出打分 Logits
        )

    def forward(self, query_emb, candidate_embs):
        """
        query_emb: (1, dim) -> (1, 2048)
        candidate_embs: (num_candidates, dim) -> (49, 2048)
        """
        num_candidates = candidate_embs.shape[0]
        query_expanded = query_emb.expand(num_candidates, -1)  # 扩展为 (49, 2048)
        
        # 拼接 [Query; Candidate]，结果形状为 (49, 3584)
        x = torch.cat([query_expanded, candidate_embs], dim=1)  # 注意这里拼接后的维度是 3584
        
        scores = self.net(x).squeeze()  # (49, 512) -> (49, 1)
        
        # Softmax 输出概率
        probs = torch.softmax(scores, dim=0)
        return probs

# ==========================================
# 5. Qwen2.5-VL 环境 (修改为使用BERTScore F1)
# ==========================================
class QwenEnvironment:
    def __init__(self):
        print(f"正在加载 Qwen2.5-VL 模型: {CONFIG['qwen_model_path']} ...")
        self.device = CONFIG["device"]
        self.pipeline = pipeline(
            CONFIG["qwen_model_path"],
            backend_config=TurbomindEngineConfig(
                session_len=8192,  # Qwen-VL 通常不需要 16k，8k 足够
                cache_max_entry_count=0.8,
                tp=1  # 如果多卡可调
            )
        )
        
        # 加载 BERT 模型用于 BERTScore 计算
        print(f"正在加载 BERT 模型用于 BERTScore 计算: {CONFIG['bert_model_path']} ...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(CONFIG["bert_model_path"])
        self.bert_model = AutoModel.from_pretrained(CONFIG["bert_model_path"]).to(self.device)
        self.bert_model.eval()
        
        # 特殊token ID
        self.special_token_ids = {
            'cls': self.bert_tokenizer.cls_token_id,
            'sep': self.bert_tokenizer.sep_token_id,
            'pad': self.bert_tokenizer.pad_token_id
        }
        
        print("BERTScore 计算器初始化完成！")
        
    def generate(self, query_data, few_shot_examples):
        """构建 Prompt 并调用模型生成 - 支持多模态ICL"""
        prompt_text = "请分析以下图片和文本是否构成讽刺。如果构成，请指出讽刺对象并解释原因。\n\n"
        
        # 存储所有图像（包括样例图像和查询图像）
        all_images = []
        
        # 1. 构建样例部分 - 为每个样例添加图像
        for i, ex in enumerate(few_shot_examples):
            # 添加样例图像占位符
            prompt_text += f"样例{i+1}:\n图片: {IMAGE_TOKEN}\n"
            # 添加样例文本和标注信息
            prompt_text += f"文本: {ex['text']}\n是否讽刺: {'是' if ex['has_sar'] else '否'}\n"
            prompt_text += f"讽刺对象: {ex['sar_obj'][0] if len(ex['sar_obj']) else '无'}\n"
            prompt_text += f"解释: {ex['sar_exp'][0] if len(ex['sar_exp']) else '无'}\n\n"
            
            # 加载样例图像
            try:
                example_image = load_image(ex['image_path'])
                all_images.append(example_image)
            except Exception as e:
                print(f"加载样例图像失败 {ex['image_path']}: {e}")
                # 使用占位图像或跳过，这里简单处理为使用查询图像
                query_image = load_image(query_data['image_path'])
                all_images.append(query_image)
        
        # 2. 构建查询部分 - 添加查询图像
        prompt_text += f"现在请分析以下内容:\n图片: {IMAGE_TOKEN}\n"
        prompt_text += f"文本: {query_data['text']}\n"
        
        # 3. 加载查询图像
        try:
            query_image = load_image(query_data['image_path'])
            all_images.append(query_image)
        except Exception as e:
            print(f"加载查询图像失败 {query_data['image_path']}: {e}")
            # 使用空白图像作为fallback
            blank_image = Image.new('RGB', (224, 224), (255, 255, 255))
            all_images.append(blank_image)
        
        # 4. 构建完整的prompt（不需要额外添加IMAGE_TOKEN，因为已经包含在prompt中）
        full_prompt = prompt_text
        print(f"完整的多模态prompt:\n{full_prompt}")
        print(f"图像数量: {len(all_images)}")
        
        # 5. 调用pipeline，传入所有图像
        response = self.pipeline((full_prompt, all_images))
        print(f"response: {response.text}")
        return response.text
    def parse_output(self, text):
        result = {"has_sar": "否", "sar_obj": "", "sar_exp": text}
        if "是" in text[:10]: result["has_sar"] = "是"
        if "讽刺对象" in text:
            try: result["sar_obj"] = text.split("讽刺对象")[1].split("\n")[0].replace(":", "").replace("：", "").strip()
            except: pass
        return result

     # 计算复合奖励
    def compute_reward(self, prediction, ground_truth, selected_indices, all_embeddings):
        truth = f"文本: {ground_truth['text']}\n是否讽刺: {'是' if ground_truth['has_sar'] else '否'}\n讽刺对象: {ground_truth['sar_obj'][0] if len(ground_truth['sar_obj']) else '无'}\n解释: {ground_truth['sar_exp'][0] if len(ground_truth['sar_exp']) else '无'}"
        print(f"Ground Truth:\n{truth}")
        """计算复合奖励"""
        reward = 0.0
        format_reward = 0.0
        fenlei_reward = 0.0
        obj_exp_reward = 0.0
        # 1. 校验预测格式是否正确 (是否讽刺，讽刺对象，解释)
        is_valid_format, pred_sar, pred_obj, pred_exp = self.validate_prediction_format(prediction)
        # print(f"各项解析结果：是否讽刺：{pred_sar}，讽刺对象：{pred_obj},讽刺解释：{pred_exp}")
        if not is_valid_format:
            print("预测格式错误，返回较低奖励")
            return -1.0  # 格式错误，返回低奖励

        # 2. 分类准确率 (最重要)
        if not ground_truth["has_sar"]:
            # 真实值为"否"，即无讽刺
            if pred_sar.strip() == "否":
                # 预测为"否"，正确
                reward += 1.0
                fenlei_reward = 1.0
            else:
                # 预测为"是"，错误
                reward -= 1.0
                fenlei_reward = -1.0
                
            # 讽刺对象和解释的打分（没有讽刺时）
            obj_exp_reward = self.compute_no_sar_reward(pred_obj, pred_exp)
            reward += obj_exp_reward

        else:
            # 真实值为"是"，即讽刺
            if pred_sar.strip() == "是":
                # 预测为"是"，正确
                reward += 1.0
                fenlei_reward = 1.0
            else:
                # 预测为"否"，错误
                reward -= 1.0
                fenlei_reward = -1.0
         
            # # 讽刺对象一致性奖励 - 使用BERTScore F1
            # obj_exp_reward = self.compute_sar_object_and_exp_similarity(ground_truth, pred_obj, pred_exp)
            # reward += obj_exp_reward

        print(f"格式奖励:{format_reward},分类奖励: {fenlei_reward},对象和解释相似度奖励: {obj_exp_reward}, 总奖励: {reward}")
        return reward
    
    def compute_sar_object_and_exp_similarity(self, ground_truth, pred_obj, pred_exp):
        """计算讽刺对象和解释的相似度（使用BERTScore F1）"""
        reward = 0.0
        
        # 计算讽刺对象相似度 - 使用BERTScore F1
        if pred_obj and ground_truth["sar_obj"]:
            # print("strike object:", ground_truth["sar_obj"])
            # print("predicted object:", pred_obj)
            similarity = self.calculate_bertscore_similarity(ground_truth["sar_obj"][0], pred_obj)
            print(f"strike object BERTScore F1: {similarity:.4f}")
            reward += similarity * 0.5  # 对象相似度权重为0.5

        # 计算讽刺解释相似度 - 使用BERTScore F1
        if pred_exp and ground_truth["sar_exp"]:
            # print("strike explanation:", ground_truth["sar_exp"])
            # print("predicted explanation:", pred_exp)
            similarity = self.calculate_bertscore_similarity(ground_truth["sar_exp"][0], pred_exp)
            print(f"strike explanation BERTScore F1: {similarity:.4f}")
            reward += similarity * 0.5  # 解释相似度权重为1.0
        
        # 限制奖励范围在[-1, 1]之间
        reward = max(min(reward, 1.0), 0.0)
        return reward

    def compute_no_sar_reward(self, pred_obj, pred_exp):
        """计算没有讽刺时的奖励"""
        reward = 0.0
        
        # 如果预测讽刺对象和解释为 '无'，则认为模型判断正确，给正奖励
        if (pred_obj == '无' or pred_obj == '无讽刺对象') and (pred_exp == '无解释' or pred_exp == '无'):
            reward += 1.0  # 给正奖励
        elif (pred_obj == '无' and pred_obj == '无讽刺对象') or (pred_exp == '无解释' and pred_exp == '无'):
            reward += 0.5  # 给部分奖励
        # 如果预测了不符合的讽刺对象或者解释，则惩罚
        else:
            reward = 0.0  # 无额外奖励
        return reward

    def calculate_bertscore_similarity(self, text1, text2):
        """
        使用BERTScore F1计算两个文本的相似度
        
        参数:
            text1: 参考文本
            text2: 预测文本
        
        返回:
            BERTScore F1 分数 (0-1范围)
        """
        if not text1 or not text2:
            return 0.0
        
        try:
            with torch.no_grad():
                # 转换为字符串并清理
                text1 = str(text1).strip()
                text2 = str(text2).strip()
                
                if not text1 or not text2:
                    return 0.0
                
                # 编码文本
                inputs1 = self.bert_tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                inputs2 = self.bert_tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                
                # 获取token embeddings
                outputs1 = self.bert_model(**inputs1, output_hidden_states=True)
                outputs2 = self.bert_model(**inputs2, output_hidden_states=True)
                
                # 使用最后一层隐藏状态
                embeddings1 = outputs1.last_hidden_state[0]  # [seq_len, hidden_size]
                embeddings2 = outputs2.last_hidden_state[0]  # [seq_len, hidden_size]
                
                # 获取token IDs
                token_ids1 = inputs1['input_ids'][0].cpu().numpy()
                token_ids2 = inputs2['input_ids'][0].cpu().numpy()
                
                # 过滤掉特殊token
                mask1 = ~np.isin(token_ids1, [self.special_token_ids['cls'], 
                                            self.special_token_ids['sep'], 
                                            self.special_token_ids['pad']])
                mask2 = ~np.isin(token_ids2, [self.special_token_ids['cls'], 
                                            self.special_token_ids['sep'], 
                                            self.special_token_ids['pad']])
                
                # 应用mask
                embeddings1 = embeddings1[mask1]
                embeddings2 = embeddings2[mask2]
                
                if len(embeddings1) == 0 or len(embeddings2) == 0:
                    return 0.0
                
                # 归一化embeddings
                embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
                embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
                
                # 计算余弦相似度矩阵 [len1, len2]
                similarity_matrix = torch.matmul(embeddings1, embeddings2.transpose(0, 1))
                
                # 计算精确率（Precision）: 对于预测文本中的每个token，找到参考文本中最相似的token
                max_sim_for_pred = torch.max(similarity_matrix, dim=0)[0]  # [len2]
                precision = torch.mean(max_sim_for_pred).item()
                
                # 计算召回率（Recall）: 对于参考文本中的每个token，找到预测文本中最相似的token
                max_sim_for_ref = torch.max(similarity_matrix, dim=1)[0]  # [len1]
                recall = torch.mean(max_sim_for_ref).item()
                
                # 计算F1分数
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                
                # 确保返回值在[0, 1]范围内
                return max(0.0, min(1.0, f1))
                
        except Exception as e:
            print(f"计算BERTScore相似度时出错: {e}")
            return 0.0

    def validate_prediction_format(self, prediction):
        """
        校验预测格式是否正确，正则提取 '是否讽刺'、'讽刺对象' 和 '解释'。
        如果格式正确，返回 (True, 是否讽刺, 讽刺对象, 解释)，否则返回 (False, None, None, None)。
        """
        sar_pattern = r"是否讽刺: (是|否)"
        obj_pattern = r"讽刺对象: (.+)"
        exp_pattern = r"解释: (.+)"
        
        # 正则匹配
        sar_match = re.search(sar_pattern, prediction)
        obj_match = re.search(obj_pattern, prediction)
        exp_match = re.search(exp_pattern, prediction)
        
        # 检查是否所有部分都存在
        if sar_match and obj_match and exp_match:
            sar = sar_match.group(1)
            obj = obj_match.group(1)
            exp = exp_match.group(1)
            return True, sar, obj, exp
        else:
            # 格式不对
            return False, None, None, None
            
# ==========================================
# 6. 主程序 (Training Loop)
# ==========================================
def main():
    # 1. 初始化
    encoder = MultimodalEncoder()
    train_dataset = RAGDataset(CONFIG["data_path"], encoder, is_training=True)
    qwen_env = QwenEnvironment()
    
    # 初始化策略网络
    policy_net = PolicyNetwork(input_dim=CONFIG["embedding_dim"], hidden_dim=CONFIG["hidden_dim"]).to(CONFIG["device"])
    optimizer = optim.Adam(policy_net.parameters(), lr=CONFIG["learning_rate"])
    
    print("\n=== 开始 REINFORCE-RAG 训练 (使用BERTScore F1) ===")
    
    for epoch in range(CONFIG["epochs"]):
        epoch_rewards = []
        epoch_loss = 0
        
        # 随机打乱训练数据
        indices = list(range(len(train_dataset.data)))
        np.random.shuffle(indices)
        
        for step, idx in enumerate(indices):
            # --- Step 1: 获取状态 (Query) ---
            query_data = train_dataset.get_item(idx)
            query_emb = train_dataset.embeddings[idx] # (dim, )
            
            # --- Step 2: 检索候选 (Environment) ---
            # 训练时排除掉自己 (ID相同)
            candidate_indices_raw, candidate_embs_raw = train_dataset.search_candidates(query_emb)
            valid_mask = [train_dataset.get_item(ci)["image_path"] != query_data["image_path"] for ci in candidate_indices_raw]
            
            # 确保有候选
            candidate_indices = candidate_indices_raw[valid_mask]
            candidate_embs = candidate_embs_raw[valid_mask]
            
            if len(candidate_indices) < CONFIG["k_shots"]: 
                continue # 候选不足，跳过
            
            # 转换为 Tensor
            q_tensor = torch.FloatTensor(query_emb).unsqueeze(0).to(CONFIG["device"])
            c_tensor = torch.FloatTensor(candidate_embs).to(CONFIG["device"])
            
            # --- Step 3: 策略网络前向 (Policy Forward) ---
            probs = policy_net(q_tensor, c_tensor)
            
            # --- Step 4: 动作采样 (Action Sampling) ---
            m = Categorical(probs)
            # 简单采样 k 个 (注意：Categorical.sample 可能重复，实际使用中可以加 Mask 处理)
            # 这里 REINFORCE 假设每次选择是独立的
            actions = m.sample((CONFIG["k_shots"],))
            
            # 获取选中的真实数据索引
            selected_db_indices = [candidate_indices[a.item()] for a in actions]
            selected_examples = [train_dataset.get_item(i) for i in selected_db_indices]
            
            # --- Step 5: 环境交互 (Environment Interaction) ---
            # 大模型生成结果
            prediction = qwen_env.generate(query_data, selected_examples)
            
            # 计算奖励 - 现在使用BERTScore F1
            reward = qwen_env.compute_reward(prediction, query_data, selected_db_indices, train_dataset.embeddings)
            epoch_rewards.append(reward)
            
            # --- Step 6: 策略更新 (REINFORCE Update) ---
            # Loss = -log_prob * reward
            # 我们希望最大化 Reward，即最小化 -Reward
            # 这里的 log_prob 是选中这 k 个样本的联合概率的对数 (假设独立则为 log_prob 之和或均值)
            log_probs = m.log_prob(actions)
            loss = -log_probs.mean() * reward 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Reward: {reward:.2f} | Loss: {loss.item():.4f}")
                
        avg_reward = np.mean(epoch_rewards)
        print(f"Epoch {epoch+1} 完成 | 平均奖励: {avg_reward:.4f} | 总 Loss: {epoch_loss:.4f}")
        
        # 保存模型
        torch.save(policy_net.state_dict(), f"./qwen_mlp_bert/policy_net_epoch_{epoch+1}_bertscore.pth")

    print("训练结束！")

if __name__ == "__main__":
    main()
    