import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

def char_tokenize(text):
    """字符级别分词（移除所有空格干扰）"""
    return " ".join(list(''.join(str(text).strip().split())))

def select_best_ref(hyp, refs):
    """
    为单个预测选择最佳参考文本
    :param hyp: 模型生成解释 (字符串)
    :param refs: 人工标注解释列表 (可能为空)
    :return: 最佳参考文本
    """
    if not refs or not hyp:
        return ""
    
    # 简化版：选择第一个非空参考（因为BERTScore会自己找最佳匹配）
    for ref in refs:
        if ref and str(ref).strip():
            return str(ref)
    
    return ""

class BERTScoreCalculator:
    """BERTScore F1计算器"""
    
    def __init__(self, model_path="bert-base-chinese", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"加载BERT模型，使用设备: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
        # 特殊token ID
        self.special_token_ids = {
            'cls': self.tokenizer.cls_token_id,
            'sep': self.tokenizer.sep_token_id,
            'pad': self.tokenizer.pad_token_id
        }
    
    def calculate_bertscore_f1(self, reference, candidate):
        """
        计算标准BERTScore F1分数
        """
        if not reference or not candidate:
            return 0.0
        
        with torch.no_grad():
            try:
                # 编码文本
                ref_inputs = self.tokenizer(reference, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                cand_inputs = self.tokenizer(candidate, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                
                # 获取token embeddings
                ref_outputs = self.model(**ref_inputs, output_hidden_states=True)
                cand_outputs = self.model(**cand_inputs, output_hidden_states=True)
                
                # 使用最后一层隐藏状态
                ref_embeddings = ref_outputs.last_hidden_state[0]  # [seq_len, hidden_size]
                cand_embeddings = cand_outputs.last_hidden_state[0]  # [seq_len, hidden_size]
                
                # 获取token IDs
                ref_token_ids = ref_inputs['input_ids'][0].cpu().numpy()
                cand_token_ids = cand_inputs['input_ids'][0].cpu().numpy()
                
                # 过滤掉特殊token
                ref_mask = ~np.isin(ref_token_ids, [self.special_token_ids['cls'], 
                                                  self.special_token_ids['sep'], 
                                                  self.special_token_ids['pad']])
                cand_mask = ~np.isin(cand_token_ids, [self.special_token_ids['cls'], 
                                                     self.special_token_ids['sep'], 
                                                     self.special_token_ids['pad']])
                
                # 应用mask
                ref_embeddings = ref_embeddings[ref_mask]
                cand_embeddings = cand_embeddings[cand_mask]
                
                if len(ref_embeddings) == 0 or len(cand_embeddings) == 0:
                    return 0.0
                
                # 归一化embeddings
                ref_embeddings = torch.nn.functional.normalize(ref_embeddings, p=2, dim=1)
                cand_embeddings = torch.nn.functional.normalize(cand_embeddings, p=2, dim=1)
                
                # 计算余弦相似度矩阵 [ref_len, cand_len]
                similarity_matrix = torch.matmul(ref_embeddings, cand_embeddings.transpose(0, 1))
                
                # 计算精确率（Precision）
                if similarity_matrix.size(0) == 0 or similarity_matrix.size(1) == 0:
                    return 0.0
                
                max_sim_for_cand = torch.max(similarity_matrix, dim=0)[0]  # [cand_len]
                precision = torch.mean(max_sim_for_cand).item()
                
                # 计算召回率（Recall）
                max_sim_for_ref = torch.max(similarity_matrix, dim=1)[0]  # [ref_len]
                recall = torch.mean(max_sim_for_ref).item()
                
                # 计算F1分数
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * (precision * recall) / (precision + recall)
                
                return f1
                
            except Exception as e:
                print(f"计算BERTScore时出错: {e}")
                return 0.0

def compute_generation_metrics(tp_refs_list, tp_hyps, bert_calculator):
    """
    为TP样本计算生成质量指标（只计算BLEU-4和BERTScore F1）
    :param tp_refs_list: 每个TP样本的人工解释列表 [[ref1, ref2, ...], ...]
    :param tp_hyps: 模型生成解释列表 [hyp1, hyp2, ...]
    :param bert_calculator: BERTScore计算器实例
    """
    if not tp_refs_list or not tp_hyps:
        return {"BLEU-4": 0, "BERTScore-F1": 0, "样本数": 0}
    
    smoothie = SmoothingFunction().method4
    
    bleu_scores = []
    bert_scores = []
    
    # 为每个样本选择最佳参考
    best_refs = []
    for refs, hyp in zip(tp_refs_list, tp_hyps):
        best_ref = select_best_ref(hyp, refs)
        best_refs.append(best_ref)
    
    # 计算生成质量指标
    for ref, hyp in tqdm(zip(best_refs, tp_hyps), total=len(best_refs), desc="计算BLEU-4和BERTScore"):
        if not ref or not hyp:
            continue
            
        ref_char = char_tokenize(ref)
        hyp_char = char_tokenize(hyp)
        
        # BLEU-4
        ref_tokens = [ref_char.split()]
        hyp_tokens = hyp_char.split()
        
        # 动态调整权重处理短文本
        if len(hyp_tokens) < 4:
            weights = (0.5, 0.5) if len(hyp_tokens) >= 2 else (1.0,)
            bleu = sentence_bleu(
                ref_tokens,
                hyp_tokens,
                weights=weights,
                smoothing_function=smoothie
            )
        else:
            bleu = sentence_bleu(
                ref_tokens,
                hyp_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothie
            )
        bleu_scores.append(bleu)
        
        # BERTScore F1
        bert_f1 = bert_calculator.calculate_bertscore_f1(ref, hyp)
        bert_scores.append(bert_f1)
    
    return {
        "BLEU-4": np.mean(bleu_scores) if bleu_scores else 0,
        "BERTScore-F1": np.mean(bert_scores) if bert_scores else 0,
        "样本数": len(bleu_scores)
    }

def compute_binary_metrics(label_has_sar, pred_has_sar):
    """基于has_sar字段计算二元决策指标"""
    tp = sum(1 for l, p in zip(label_has_sar, pred_has_sar) if l and p)
    tn = sum(1 for l, p in zip(label_has_sar, pred_has_sar) if not l and not p)
    fp = sum(1 for l, p in zip(label_has_sar, pred_has_sar) if not l and p)
    fn = sum(1 for l, p in zip(label_has_sar, pred_has_sar) if l and not p)
    
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,  # label有讽刺 & 模型预测有讽刺
        "TN": tn,  # label无讽刺 & 模型预测无讽刺
        "FP": fp,  # label无讽刺 & 模型错误预测有讽刺
        "FN": fn   # label有讽刺 & 模型未预测到
    }

def main(json_path, bert_calculator):
    # 读取JSON数据
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取关键字段
    label_has_sar = []
    pred_has_sar = []
    sar_exp_lists = []  # 每个样本的参考解释列表
    llm_exps = []       # 模型生成解释
    
    for item in data:
        # 安全获取字段
        has_sar = bool(item.get('has_sar', False))
        llm_has_sar = bool(item.get('has_sar_llm', False))
        
        # 处理sar_exp: 确保是列表格式
        sar_exp = item.get('sar_exp', [])
        if not isinstance(sar_exp, list):
            sar_exp = [sar_exp] if sar_exp else []
        
        # 处理llm_exp: 确保是字符串
        llm_exp = str(item.get('sar_exp_llm', '')).strip() or ""
        
        label_has_sar.append(has_sar)
        pred_has_sar.append(llm_has_sar)
        sar_exp_lists.append(sar_exp)
        llm_exps.append(llm_exp)
    
    # 1. 计算二元决策指标
    binary_metrics = compute_binary_metrics(label_has_sar, pred_has_sar)
    
    # 2. 筛选TP样本 (has_sar=True 且 llm_has_sar=True)
    tp_indices = [
        i for i, (l, p) in enumerate(zip(label_has_sar, pred_has_sar))
        if l and p
    ]
    
    tp_refs_list = [sar_exp_lists[i] for i in tp_indices]
    tp_hyps = [llm_exps[i] for i in tp_indices]
    
    print(f"\n🔍 严格筛选TP样本:")
    print(f"  - 标注有讽刺(has_sar=True) 且 模型预测有讽刺(llm_has_sar=True): {len(tp_indices)} 个样本")
    
    # 3. 计算生成质量指标 (只计算BLEU-4和BERTScore F1)
    gen_metrics = compute_generation_metrics(tp_refs_list, tp_hyps, bert_calculator)
    
    # 4. 生成详细报告
    total = len(data)
    print("\n" + "="*70)
    print(f"讽刺解释生成综合评估报告 (BLEU-4 + BERTScore F1)")
    print("="*70)
    
    # 二元决策指标
    print(f"\n🎯 二元决策指标 (讽刺检测能力):")
    print(f"   Accuracy : {binary_metrics['Accuracy']:.4f}")
    print(f"   Precision: {binary_metrics['Precision']:.4f}")
    print(f"   Recall   : {binary_metrics['Recall']:.4f}")
    print(f"   F1       : {binary_metrics['F1']:.4f}")
    print(f"   TP: {binary_metrics['TP']}  FP: {binary_metrics['FP']}  FN: {binary_metrics['FN']}  TN: {binary_metrics['TN']}")
    
    # 生成质量指标
    print(f"\n📝 生成质量指标 (在{gen_metrics['样本数']}个TP样本上计算):")
    print(f"   BLEU-4     : {gen_metrics['BLEU-4']:.4f}  (n-gram精确匹配)")
    print(f"   BERTScore-F1: {gen_metrics['BERTScore-F1']:.4f}  (语义相似度)")
    
    print("\n" + "="*70)
    
    return {
        "binary_metrics": binary_metrics,
        "generation_metrics": gen_metrics,
        "file_name": os.path.basename(json_path)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='讽刺解释生成评估 (BLEU-4 + BERTScore F1)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='JSON文件目录路径')
    parser.add_argument('--bert_model_path', type=str, default="/home/google-bert/bert-base-chinese",
                        help='BERT模型路径，默认为"bert-base-chinese"')
    args = parser.parse_args()
    
    # 依赖检查
    try:
        from nltk.translate.bleu_score import sentence_bleu
        import torch
        from transformers import AutoTokenizer, AutoModel
    except ImportError:
        print("❗ 依赖缺失，请安装: pip install nltk torch transformers tqdm numpy")
        exit(1)
    
    # 初始化BERTScore计算器
    bert_calculator = BERTScoreCalculator(model_path=args.bert_model_path)
    
    # 获取文件列表
    filelist = [f for f in os.listdir(args.input_dir) if f.endswith('.json')]
    if not filelist:
        print(f"⚠️  在目录 {args.input_dir} 中没有找到JSON文件")
        exit(1)
    
    print(f"📂 找到 {len(filelist)} 个JSON文件，开始评估...")
    
    # 评估所有文件
    all_results = []
    for file in filelist:
        input_file = os.path.join(args.input_dir, file)
        print(f"\n" + "="*80)
        print(f"📊 评估文件: {file}")
        print("="*80)
        result = main(input_file, bert_calculator)
        all_results.append(result)
    
    # 生成汇总报告
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("📈 评估结果汇总")
        print("="*80)
        
        for result in all_results:
            print(f"\n文件: {result['file_name']}")
            print(f"  二元决策 F1: {result['binary_metrics']['F1']:.4f}")
            print(f"  BLEU-4: {result['generation_metrics']['BLEU-4']:.4f}")
            print(f"  BERTScore-F1: {result['generation_metrics']['BERTScore-F1']:.4f}")
            print(f"  样本数: {result['generation_metrics']['样本数']}")