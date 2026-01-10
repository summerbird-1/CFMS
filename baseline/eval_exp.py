import json
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import argparse
import itertools
import os

def char_tokenize(text):
    """字符级别分词（移除所有空格干扰）"""
    return " ".join(list(''.join(str(text).strip().split())))

def select_best_ref(hyp, refs):
    """
    为单个预测选择最佳参考文本
    :param hyp: 模型生成解释 (字符串)
    :param refs: 人工标注解释列表 (可能为空)
    :return: 最佳参考文本 + 该匹配的ROUGE-L分数
    """
    if not refs or not hyp:
        return "", 0.0
    
    # 初始化ROUGE计算器
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, split_summaries=True)
    hyp_char = char_tokenize(hyp)
    
    best_score = -1
    best_ref = ""
    
    for ref in refs:
        if not ref:  # 跳过空参考
            continue
        ref_char = char_tokenize(str(ref))
        score = scorer.score(ref_char, hyp_char)['rougeL'].fmeasure
        if score > best_score:
            best_score = score
            best_ref = str(ref)
    
    return best_ref, best_score

def compute_generation_metrics(tp_refs_list, tp_hyps):
    """
    为TP样本计算生成质量指标
    :param tp_refs_list: 每个TP样本的人工解释列表 [[ref1, ref2, ...], ...]
    :param tp_hyps: 模型生成解释列表 [hyp1, hyp2, ...]
    """

    if not tp_refs_list or not tp_hyps:
        return {"BLEU-4": 0, "ROUGE-1": 0, "ROUGE-2": 0, "ROUGE-L": 0, "样本数": 0}
    
    # 初始化ROUGE计算器
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=False,
        split_summaries=True
    )
    smoothie = SmoothingFunction().method4
    
    bleu_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []
    match_scores = []  # 记录最佳匹配分数
    
    # 为每个样本选择最佳参考
    best_refs = []
    for refs, hyp in tqdm(zip(tp_refs_list, tp_hyps), total=len(tp_refs_list), desc="选择最佳参考"):
        best_ref, match_score = select_best_ref(hyp, refs)
        best_refs.append(best_ref)
        match_scores.append(match_score)
    
    # 计算生成质量指标
    for ref, hyp in zip(best_refs, tp_hyps):
        if not ref or not hyp:
            continue
            
        ref_char = char_tokenize(ref)
        hyp_char = char_tokenize(hyp)
        
        # ROUGE
        rouge_scores = scorer.score(ref_char, hyp_char)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougel_scores.append(rouge_scores['rougeL'].fmeasure)
        
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
    
    return {
        "BLEU-4": np.mean(bleu_scores) if bleu_scores else 0,
        "ROUGE-1": np.mean(rouge1_scores) if rouge1_scores else 0,
        "ROUGE-2": np.mean(rouge2_scores) if rouge2_scores else 0,
        "ROUGE-L": np.mean(rougel_scores) if rougel_scores else 0,
        "平均最佳匹配分数": np.mean(match_scores) if match_scores else 0,
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

def main(json_path):
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
        llm_has_sar = bool(item.get('llm_has_sar', False))
        
        # 处理sar_exp: 确保是列表格式
        sar_exp = item.get('sar_exp', [])
        if not isinstance(sar_exp, list):
            sar_exp = [sar_exp] if sar_exp else []
        
        # 处理llm_exp: 确保是字符串
        llm_exp = str(item.get('llm_sar_exp', '')[0]).strip() or ""
        
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
    
    # 3. 计算生成质量指标 (动态选择最佳参考)
    
    gen_metrics = compute_generation_metrics(tp_refs_list, tp_hyps)
    print(gen_metrics)
    # 4. 计算综合评分
    composite_bleu = binary_metrics["F1"] * gen_metrics["BLEU-4"]
    composite_rougel = binary_metrics["F1"] * gen_metrics["ROUGE-L"]
    
    # 5. 生成详细报告
    total = len(data)
    print("\n" + "="*70)
    print(f"讽刺解释生成综合评估报告 (多参考标注优化版)")
    print("="*70)
    
    # # 样本分布
    # print(f"\n📊 样本分布统计 (总样本: {total})")
    # print(f"   ✅ 真阳性(TP): {binary_metrics['TP']:4d}  (label有讽刺 & 模型预测有讽刺)")
    # print(f"   ✅ 真阴性(TN): {binary_metrics['TN']:4d}  (label无讽刺 & 模型预测无讽刺)")
    # print(f"   ❌ 假阳性(FP): {binary_metrics['FP']:4d}  (过度解释错误)")
    # print(f"   ❌ 假阴性(FN): {binary_metrics['FN']:4d}  (漏解释错误)")
    
    # # 二元决策指标
    # print(f"\n🔍 二元决策能力 (讽刺存在性判断):")
    # print(f"   Accuracy: {binary_metrics['Accuracy']:.4f}  (整体正确率)")
    # print(f"   Precision: {binary_metrics['Precision']:.4f}  (避免过度解释)")
    # print(f"   Recall:    {binary_metrics['Recall']:.4f}  (避免漏解释)")
    # print(f"   F1-score:  {binary_metrics['F1']:.4f}      ← 核心决策指标")
    
    # 生成质量指标
    print(f"文件名称: {json_path}")
    print(f"\n📝 生成质量指标 (在{gen_metrics['样本数']}个TP样本上计算):")
    print(f"   平均最佳匹配分数: {gen_metrics['平均最佳匹配分数']:.4f}  (ROUGE-L, 选择参考质量)")
    print(f"   BLEU-4 : {gen_metrics['BLEU-4']:.4f}  (n-gram精确匹配)")
    print(f"   ROUGE-1: {gen_metrics['ROUGE-1']:.4f}  (词级覆盖)")
    print(f"   ROUGE-2: {gen_metrics['ROUGE-2']:.4f}  (短语级覆盖)")
    print(f"   ROUGE-L: {gen_metrics['ROUGE-L']:.4f}  ← 重点关注 (语义连贯性)")
    
    # # 综合评分
    # print(f"\n⭐ 综合能力评分 (F1 × 生成质量):")
    # print(f"   BLEU-4 综合: {composite_bleu:.4f}")
    # print(f"   ROUGE-L综合: {composite_rougel:.4f}  ← 最终参考指标")
    
    print("="*70)
    
    # # 6. 保存完整结果
    # result = {
    #     "评估概述": {
    #         "总样本数": total,
    #         "使用字段": ["has_sar (人工标注)", "llm_has_sar (模型预测)", "sar_exp (人工解释列表)", "llm_exp (模型解释)"],
    #         "关键优化": "为每个样本动态选择最佳匹配的人工解释 (基于ROUGE-L)",
    #         "TP定义": "has_sar=True 且 llm_has_sar=True",
    #         "评估层级": [
    #             "1. 讽刺存在性判断 (二元分类)",
    #             "2. 解释生成质量 (仅TP样本，动态选择最佳参考)",
    #             "3. 综合能力评分 (F1 × ROUGE-L)"
    #         ]
    #     },
    #     "样本分布": {
    #         "真阳性(TP)": binary_metrics["TP"],
    #         "真阴性(TN)": binary_metrics["TN"],
    #         "假阳性(FP)": binary_metrics["FP"],
    #         "假阴性(FN)": binary_metrics["FN"]
    #     },
    #     "二元决策指标": {
    #         "Accuracy": binary_metrics["Accuracy"],
    #         "Precision": binary_metrics["Precision"],
    #         "Recall": binary_metrics["Recall"],
    #         "F1": binary_metrics["F1"]
    #     },
    #     "生成质量指标": gen_metrics,
    #     "综合评分": {
    #         "Composite_BLEU4": composite_bleu,
    #         "Composite_ROUGE-L": composite_rougel,
    #         "计算方式": "F1_score * ROUGE-L"
    #     },
    #     "评估参数": {
    #         "参考选择策略": "为每个样本选择ROUGE-L分数最高的人工解释",
    #         "分词级别": "字符级别 (character-level)",
    #         "BLEU平滑": "SmoothingFunction.method4",
    #         "ROUGE设置": "use_stemmer=False, split_summaries=True"
    #     }
    # }
    
    # # 保存结果
    # output_path = json_path.replace('.json', '_metrics.json')
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     json.dump(result, f, indent=2, ensure_ascii=False)
    # print(f"\n💾 完整评估结果已保存至: {output_path}")
    
    # # 7. 生成错误分析样本（可选）
    # if binary_metrics["FP"] > 0 or binary_metrics["FN"] > 0:
    #     error_samples = []
        
    #     # 收集FP样本 (过度解释)
    #     fp_indices = [i for i, (l, p) in enumerate(zip(label_has_sar, pred_has_sar)) if not l and p]
    #     for i in fp_indices[:3]:  # 取前3个
    #         error_samples.append({
    #             "类型": "FP (过度解释)",
    #             "原文": data[i].get("text", "N/A"),
    #             "人工标注": {
    #                 "has_sar": label_has_sar[i],
    #                 "sar_exp": sar_exp_lists[i]
    #             },
    #             "模型预测": {
    #                 "llm_has_sar": pred_has_sar[i],
    #                 "llm_exp": llm_exps[i]
    #             }
    #         })
        
    #     # 收集FN样本 (漏解释)
    #     fn_indices = [i for i, (l, p) in enumerate(zip(label_has_sar, pred_has_sar)) if l and not p]
    #     for i in fn_indices[:3]:  # 取前3个
    #         error_samples.append({
    #             "类型": "FN (漏解释)",
    #             "原文": data[i].get("text", "N/A"),
    #             "人工标注": {
    #                 "has_sar": label_has_sar[i],
    #                 "sar_exp": sar_exp_lists[i]
    #             },
    #             "模型预测": {
    #                 "llm_has_sar": pred_has_sar[i],
    #                 "llm_exp": llm_exps[i]
    #             }
    #         })
        
    #     # 收集生成质量差的TP样本
    #     if gen_metrics["样本数"] > 0:
    #         # 重新计算每个TP样本的ROUGE-L (使用最佳参考)
    #         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False, split_summaries=True)
    #         tp_scores = []
            
    #         for i, idx in enumerate(tp_indices):
    #             refs = sar_exp_lists[idx]
    #             hyp = llm_exps[idx]
    #             best_ref, _ = select_best_ref(hyp, refs)
    #             if best_ref and hyp:
    #                 score = scorer.score(
    #                     char_tokenize(best_ref), 
    #                     char_tokenize(hyp)
    #                 )['rougeL'].fmeasure
    #                 tp_scores.append((score, idx, best_ref, hyp))
            
    #         # 选择ROUGE-L最低的3个样本
    #         tp_scores.sort(key=lambda x: x[0])  # 从小到大排序
    #         for score, idx, best_ref, hyp in tp_scores[:3]:
    #             error_samples.append({
    #                 "类型": f"TP-低质量 (ROUGE-L={score:.4f})",
    #                 "原文": data[idx].get("text", "N/A"),
    #                 "人工标注": {
    #                     "has_sar": label_has_sar[idx],
    #                     "sar_exp": sar_exp_lists[idx],
    #                     "最佳匹配参考": best_ref
    #                 },
    #                 "模型预测": {
    #                     "llm_has_sar": pred_has_sar[idx],
    #                     "llm_exp": hyp
    #                 }
    #             })
        
    #     error_path = json_path.replace('.json', '_error_samples.json')
    #     with open(error_path, 'w', encoding='utf-8') as f:
    #         json.dump(error_samples, f, indent=2, ensure_ascii=False)
    #     print(f"🔍 典型错误样本已保存至: {error_path} (包含FP/FN/低质量TP示例)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='讽刺解释生成评估 (多参考标注优化版)')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='JSON文件路径，需包含: has_sar, llm_has_sar, sar_exp(列表), llm_exp')
    args = parser.parse_args()
    
    # 依赖检查
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
    except ImportError:
        print("❗ 依赖缺失，请安装: pip install rouge_score nltk tqdm numpy")
        exit(1)
    filelist = os.listdir(args.input_dir)
    for file in filelist:
        input_file = os.path.join(args.input_dir, file)
        main(input_file)
