# build_multimodal_index_bge_clip.py
import json
import numpy as np
import faiss
import os
import argparse
from PIL import Image
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel

# 初始化编码器
bge = SentenceTransformer('./BAAI/bge-large-zh-v1.5')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model.to(device)
def build_index(known_data_path):
    count = 0
    # 加载训练数据
    traindata = json.load(open(known_data_path))
    texts = []
    image_paths = []
    valid_indices = []

    for idx, item in enumerate(traindata):
        img_path = item.get('image_path')
        if not img_path:
            continue

        
        text = item['text'] + item['desc']
        print(text)
        
        texts.append(text)
        image_paths.append(img_path)
        valid_indices.append(idx)
    print(f"共有{count} 条无效样本")
    print(f"共处理 {len(texts)} 条有效样本")

    # === 提取 BGE 文本嵌入（1024 维）===
    print("提取 BGE 文本嵌入...")
    text_embs = bge.encode(texts, normalize_embeddings=True, batch_size=64)  # [N, 1024]

    # === 提取 CLIP 图像嵌入（512 维）===
    print("提取 CLIP 图像嵌入...")
    img_embs = []
    batch_size = 32
    for i in range(0, len(image_paths), batch_size):
        batch_imgs = []
        for path in image_paths[i:i+batch_size]:
            try:
                if path.startswith("http"):
                    img = Image.open(requests.get(path, stream=True).raw).convert("RGB")
                else:
                    img = Image.open(path).convert("RGB")
                batch_imgs.append(img)
            except Exception as e:
                print(f"⚠️ 图片加载失败: {path}，使用空白图替代")
                batch_imgs.append(Image.new("RGB", (224, 224), (255, 255, 255)))
        
        inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)  # [B, 512]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            img_embs.append(emb.cpu().numpy())
    img_embs = np.vstack(img_embs)  # [N, 512]

    # === 拼接多模态嵌入 ===
    multimodal_embs = np.concatenate([text_embs, img_embs], axis=1)  # [N, 1536]

    # === 构建 FAISS 索引 ===
    dim = 1536
    index = faiss.index_factory(dim, "IVF256,PQ48")  # 1536 ÷ 48 = 32，满足 PQ 要求
    index.train(multimodal_embs.astype('float32'))
    index.add(multimodal_embs.astype('float32'))
    faiss.write_index(index, "sarc_multimodal_bge_clip.index")
    np.save("valid_indices.npy", np.array(valid_indices))

    print("✅ 多模态索引已保存：sarc_multimodal_bge_clip.index")
    
def get_example(testdata,traindata,example_path,k_shots=2):
    # === 加载测试数据 ===
    testdata = json.load(open(testdata))

    # === 加载训练时保存的索引和 valid_indices ===
    index = faiss.read_index("sarc_multimodal_bge_clip.index")
    valid_indices = np.load("valid_indices.npy")  # 对应训练集中有效样本的原始索引

    # 加载原始训练数据（用于获取检索到的样本内容）
    traindata = json.load(open(traindata))

    # === 参数设置 ===
    k = k_shots  # 检索 top-k 个相似样本
    batch_size = 32

    # 存储每条测试样本的检索结果
    retrieved_examples = []

    # === 预处理测试数据 ===
    test_texts = []
    test_image_paths = []
    test_valid_mask = []  # 标记哪些测试样本有效（有图片）

    for item in testdata:
        img_path = item.get('image_path')
        if not img_path:
            test_valid_mask.append(False)
            continue

        
        text = item['text'] + item['desc']
            
        test_texts.append(text)
        test_image_paths.append(img_path)
        test_valid_mask.append(True)

    print(f"共处理 {len(test_texts)} 条有效测试样本")

    # === 提取测试样本的多模态嵌入 ===
    print("提取测试样本 BGE 文本嵌入...")
    test_text_embs = bge.encode(test_texts, normalize_embeddings=True, batch_size=64)  # [M, 1024]

    print("提取测试样本 CLIP 图像嵌入...")
    test_img_embs = []
    for i in range(0, len(test_image_paths), batch_size):
        batch_imgs = []
        for path in test_image_paths[i:i+batch_size]:
            try:
                if path.startswith("http"):
                    img = Image.open(requests.get(path, stream=True).raw).convert("RGB")
                else:
                    img = Image.open(path).convert("RGB")
                batch_imgs.append(img)
            except Exception as e:
                print(f"⚠️ 测试图片加载失败: {path}，使用空白图替代")
                batch_imgs.append(Image.new("RGB", (224, 224), (255, 255, 255)))
        
        inputs = clip_processor(images=batch_imgs, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            test_img_embs.append(emb.cpu().numpy())

    test_img_embs = np.vstack(test_img_embs)  # [M, 512]
    test_multimodal_embs = np.concatenate([test_text_embs, test_img_embs], axis=1).astype('float32')  # [M, 1536]

    # === 批量检索 ===
    print(f"在索引中检索 top-{k} 相似样本...")
    distances, indices = index.search(test_multimodal_embs, k)  # indices: [M, k]

    # === 构建检索结果 ===
    all_retrieved = []
    test_idx = 0
    for i, item in enumerate(testdata):
        if not test_valid_mask[i]:
            # 无效样本，无检索结果
            all_retrieved.append([])
            continue

        # 获取 top-k 训练样本的原始索引
        train_idxs = valid_indices[indices[test_idx]].tolist()
        examples = []
        for tid in train_idxs:
            ex = traindata[tid]
            # 可选：只保留必要字段，如 text, image_path, label 等
            examples.append({
                "text": ex["text"],
                "image_path": ex.get("image_path", ""),
                "has_sar": ex["has_sar"],
                "sar_obj": ex.get("sar_obj", []),
                "sar_exp": ex.get("sar_exp", [])
            })
        all_retrieved.append(examples)
        test_idx += 1

    # === 保存检索结果（可选）===
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(all_retrieved, f, ensure_ascii=False, indent=2)

    print(f"✅ 检索完成，结果已保存至 {example_path}")
if __name__ == "__main__":
    
     # 创建参数解析器
    parser = argparse.ArgumentParser(description='构建多模态RAG索引并检索相似样本')
    
    # 添加参数
    parser.add_argument('--traindata', type=str, default="../xhsdata/traindata_zh.json",
                        help='训练数据路径')
    parser.add_argument('--testdata', type=str, default="../xhsdata/testdata_zh.json",
                        help='测试数据路径')
    parser.add_argument('--output', type=str, default="retrieved_examples_2shot.json",
                        help='检索结果输出路径')
    parser.add_argument('--k_shots', type=int, default=2,
                        help='检索的相似样本数量')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查是否需要构建索引
    if not os.path.exists("sarc_multimodal_bge_clip.index") or not os.path.exists("valid_indices.npy"):
        print("正在构建索引...")
        build_index(args.traindata)
    
    # 检索相似样本
    print("正在检索相似样本...")
    get_example(args.testdata, args.traindata, args.output, args.k_shots)