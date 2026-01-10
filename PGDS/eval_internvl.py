#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理脚本：使用训练好的策略网络为测试集生成预测
只需要生成预测结果，不进行评估
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

# 导入必要的类（假设这些类与训练时相同）
from bert_mian_internvl import (
    MultimodalEncoder,
    RAGDataset,
    PolicyNetwork,
    QwenEnvironment
)


def load_policy_model(model_path, input_dim, hidden_dim, device):
    """加载训练好的策略网络"""
    print(f"加载策略网络: {model_path}")
    policy_net = PolicyNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    if os.path.exists(model_path):
        # 加载完整的模型状态
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 检查点包含模型状态
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            print(f"从检查点加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # 直接是模型状态
            policy_net.load_state_dict(checkpoint)
            print("加载模型成功")
    else:
        raise FileNotFoundError(f"模型文件 {model_path} 不存在!")
    
    policy_net.eval()  # 设置为评估模式
    return policy_net


def convert_to_serializable(obj):
    """递归地将非JSON可序列化对象转换为可序列化格式"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # 对于其他类型，尝试转换为字符串
        try:
            return str(obj)
        except:
            return f"Unserializable object: {type(obj).__name__}"


def generate_predictions(policy_net, train_dataset, test_dataset, qwen_env, device, k_shots=1):
    """
    为测试集生成预测
    
    参数:
        policy_net: 训练好的策略网络
        train_dataset: 训练数据集（候选池）
        test_dataset: 测试数据集
        qwen_env: Qwen环境
        device: 设备
        k_shots: few-shot例子数量
    
    返回:
        predictions: 预测结果列表
    """
    predictions = []
    
    print(f"\n开始为 {len(test_dataset.data)} 个测试样本生成预测...")
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset.data)), desc="生成预测"):
            # 获取测试查询
            query_data = test_dataset.get_item(idx)
            query_emb = test_dataset.embeddings[idx]
            
            # 从训练集中检索候选
            candidate_indices_raw, candidate_embs_raw = train_dataset.search_candidates(query_emb)
            
            # 排除测试样本自身（如果与训练集有重叠）
            valid_mask = [True] * len(candidate_indices_raw)
            for i, ci in enumerate(candidate_indices_raw):
                try:
                    candidate_item = train_dataset.get_item(ci)
                    # 如果图像路径相同，排除
                    if candidate_item.get("image_path") == query_data.get("image_path"):
                        valid_mask[i] = False
                except:
                    pass
            
            if sum(valid_mask) == 0:
                print(f"警告：测试样本 {idx} 没有有效候选，使用所有候选")
                valid_mask = [True] * len(candidate_indices_raw)
            
            candidate_indices = candidate_indices_raw[valid_mask]
            candidate_embs = candidate_embs_raw[valid_mask]
            
            if len(candidate_indices) < k_shots:
                print(f"警告：测试样本 {idx} 候选不足 ({len(candidate_indices)} < {k_shots})，使用所有候选")
                # 如果还是不够，使用所有候选（即使重复）
                candidate_indices = candidate_indices_raw
                candidate_embs = candidate_embs_raw
            
            # 转换为Tensor
            q_tensor = torch.FloatTensor(query_emb).unsqueeze(0).to(device)
            c_tensor = torch.FloatTensor(candidate_embs).to(device)
            
            # 策略网络前向（评估模式下选择概率最高的k个）
            probs = policy_net(q_tensor, c_tensor)
            
            # 选择概率最高的k个候选（贪心策略）
            topk_values, topk_indices = torch.topk(probs, min(k_shots, len(probs)))
            
            # 获取选中的例子
            selected_db_indices = [candidate_indices[a.item()] for a in topk_indices]
            selected_examples = [train_dataset.get_item(i) for i in selected_db_indices]
            
            # 生成预测
            try:
                prediction = qwen_env.generate(query_data, selected_examples)
                
                # 解析预测结果
                parsed_pred = qwen_env.parse_output(prediction)
            except Exception as e:
                print(f"生成预测时出错 (样本 {idx}): {e}")
                # 创建空预测
                parsed_pred = {
                    "has_sar": "未知",
                    "sar_obj": "",
                    "sar_exp": ""
                }
                prediction = ""
            
            # 记录预测结果
            pred_record = {
                "test_sample_id": idx,
                "test_sample_text": query_data.get("text", ""),
                "test_sample_image": query_data.get("image_path", ""),
                "has_sar": query_data.get("has_sar", False),
                "sar_obj": query_data.get("sar_obj", ""),
                "sar_exp": query_data.get("sar_exp", ""),
                "selected_fewshot_indices": selected_db_indices,
                "selected_fewshot_examples": [
                    {
                        "text": ex.get("text", ""),
                        "image_path": ex.get("image_path", ""),
                        "has_sar": ex.get("has_sar", False)
                    }
                    for ex in selected_examples
                ],
                "llm_res": prediction,
                "prediction_parsed": parsed_pred,
                "policy_network_probs": probs.cpu().numpy().tolist() if torch.is_tensor(probs) else probs,
                "selected_probs": topk_values.cpu().numpy().tolist() if torch.is_tensor(topk_values) else topk_values
            }
            
            predictions.append(pred_record)
    
    return predictions


def save_predictions(predictions, output_dir, config_info):
    """
    保存预测结果到文件
    
    参数:
        predictions: 预测结果列表
        output_dir: 输出目录
        config_info: 配置信息
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换所有预测结果为JSON可序列化格式
    serializable_predictions = []
    for pred in predictions:
        serializable_pred = convert_to_serializable(pred)
        serializable_predictions.append(serializable_pred)
    
    # 保存完整的预测结果
    output_file = os.path.join(output_dir, "predictions.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serializable_predictions, f, ensure_ascii=False, indent=2)
    
    print(f"预测结果已保存到: {output_file}")
    
    # 保存一个简化的版本（便于查看）
    simplified_predictions = []
    for pred in serializable_predictions:
        simplified = {
            "test_sample_id": pred["test_sample_id"],
            "test_sample_text": pred["test_sample_text"][:100] + "..." if len(pred["test_sample_text"]) > 100 else pred["test_sample_text"],
            "ground_truth_has_sar": pred.get("has_sar", False),
            "predicted_has_sar": pred["prediction_parsed"].get("has_sar", "未知"),
            "selected_fewshot_count": len(pred["selected_fewshot_examples"])
        }
        simplified_predictions.append(simplified)
    
    simplified_file = os.path.join(output_dir, "predictions_simplified.json")
    with open(simplified_file, "w", encoding="utf-8") as f:
        json.dump(simplified_predictions, f, ensure_ascii=False, indent=2)
    
    # 保存配置信息
    serializable_config = convert_to_serializable(config_info)
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(serializable_config, f, ensure_ascii=False, indent=2)
    
    # 生成一个统计报告
    total_count = len(serializable_predictions)
    if total_count > 0:
        # 统计预测结果
        prediction_counts = {}
        for pred in serializable_predictions:
            pred_sar = pred["prediction_parsed"].get("has_sar", "未知")
            prediction_counts[pred_sar] = prediction_counts.get(pred_sar, 0) + 1
        
        # 统计真实标签
        ground_truth_counts = {}
        for pred in serializable_predictions:
            gt_sar = pred.get("has_sar", False)
            key = "是" if gt_sar else "否"
            ground_truth_counts[key] = ground_truth_counts.get(key, 0) + 1
        
        stats = {
            "total_predictions": total_count,
            "prediction_distribution": prediction_counts,
            "ground_truth_distribution": ground_truth_counts,
            "fewshot_examples_per_sample": serializable_config.get("k_shots", 1),
            "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        serializable_stats = convert_to_serializable(stats)
        stats_file = os.path.join(output_dir, "statistics.json")
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
        
        print(f"\n统计信息:")
        print(f"  总测试样本数: {total_count}")
        print(f"  预测分布: {prediction_counts}")
        print(f"  真实分布: {ground_truth_counts}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用训练好的策略网络生成预测")
    parser.add_argument("--train_data", type=str, default="./traindata.json",
                       help="训练数据路径")
    parser.add_argument("--test_data", type=str, default="./testdata.json",
                       help="测试数据路径")
    parser.add_argument("--model_path", type=str, required=True,
                       help="训练好的策略网络模型路径")
    parser.add_argument("--output_dir", type=str, default="./intern_inference_results3",
                       help="输出目录")
    parser.add_argument("--k_shots", type=int, default=1,
                       help="few-shot例子数量")
    parser.add_argument("--device", type=str, default=None,
                       help="设备 (cuda/cpu)，默认自动选择")
    
    args = parser.parse_args()
    
    # 配置参数
    CONFIG = {
        "train_data_path": args.train_data,
        "test_data_path": args.test_data,
        "model_path": args.model_path,
        "output_dir": args.output_dir,
        "k_shots": args.k_shots,
        "qwen_model_path": "/home/LLaMA-Factory/OpenGVLab/InternVL2_5-8B-hf",
        "bge_model_path": "/home/BAAI/bge-large-zh-v1.5",
        "clip_model_path": "/home/openai/clip-vit-base-patch32",
        "embedding_dim": 3584,
        "hidden_dim": 512,
        "top_k_candidates": 50,
    }
    
    # 设置设备
    if args.device:
        CONFIG["device"] = args.device
    else:
        CONFIG["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"设备: {CONFIG['device']}")
    print(f"配置: {CONFIG}")
    
    # 创建时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_path}")
    
    try:
        # 1. 加载编码器
        print("\n1. 加载多模态编码器...")
        encoder = MultimodalEncoder()
        
        # 2. 加载数据集
        print("\n2. 加载数据集...")
        train_dataset = RAGDataset(CONFIG["train_data_path"], encoder, is_training=True)
        test_dataset = RAGDataset(CONFIG["test_data_path"], encoder, is_training=False)
        
        print(f"  训练集大小: {len(train_dataset.data)}")
        print(f"  测试集大小: {len(test_dataset.data)}")
        
        # 3. 加载训练好的策略网络
        policy_net = load_policy_model(
            model_path=CONFIG["model_path"],
            input_dim=CONFIG["embedding_dim"],
            hidden_dim=CONFIG["hidden_dim"],
            device=CONFIG["device"]
        )
        
        # 4. 加载Qwen环境
        print("\n4. 加载Qwen环境...")
        qwen_env = QwenEnvironment()
        
        # 5. 生成预测
        predictions = generate_predictions(
            policy_net=policy_net,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            qwen_env=qwen_env,
            device=CONFIG["device"],
            k_shots=CONFIG["k_shots"]
        )
        
        # 6. 保存预测结果
        save_predictions(
            predictions=predictions,
            output_dir=output_dir,
            config_info=CONFIG
        )
        
        print(f"\n推理完成！结果保存在: {output_dir}")
        
    except Exception as e:
        print(f"推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())