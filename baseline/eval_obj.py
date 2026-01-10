import argparse
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json
from openai import OpenAI

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key='sk-K8vDjmVIPryP4htiwsV6BmSNR0YwnoqDB27Z1bLq0ZeL4r0a',
)

def deepseekapi(obj_label, obj_pred):
    print("+++" * 20)
    prompt = f"""你是一个评测专家。请判断模型预测的讽刺对象是否正确。
**真实讽刺对象**：{obj_label}  
**模型预测的讽刺对象**：{obj_pred}
**判断标准**：
1. 预测对象只要语义相近即视为正确，不要求字面完全一致。
2. 如果真实讽刺对象包含多个，只要预测对象与其中任意一个语义相近，即视为正确。
请严格按以下格式回答：  
如果是 → 输出“是”  
如果否 → 输出“否”  
不要输出任何其他内容。"""
    print(f"调用ds生成文本的请求为：\n{prompt}")
    
    # return "是"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ]
    )
    print("+++" * 20)
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def process_files(input_dir, output_dir):
    """
    处理输入目录中的所有JSON文件
    
    Args:
        input_dir (str): 输入目录路径
        output_dir (str): 输出目录路径
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    filelist = [file for file in os.listdir(input_dir) if not file.endswith('2.json')]
    
    for file in filelist:
        print(f"Processing {file}...")
        file_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        # 加载数据
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 处理每条数据
        for item in data:
            try:
                has_sar = item['has_sar']
                if not has_sar:
                    continue
                if not item['llm_has_sar'] or not item['llm_sar_obj']:
                    item['obj_flag'] = False
                else:
                    obj_label = item['sar_obj']
                    obj_pred = item['llm_sar_obj'][0]
                    ds_res = deepseekapi(obj_label, obj_pred)
                    item['ds_res'] = ds_res
                # 转换结果格式
                if '是' in ds_res:
                    item['obj_flag'] = True
                else :
                    item['obj_flag'] = False
            except Exception as e:
                print(f"{file}出错")
                print(f"{item},{e}")
        
        # 保存处理后的数据
        with open(output_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"{file} processed")

def calculate_metrics(output_dir):
    """
    仅计算讽刺对象识别性能指标（在真实有讽刺的样本上）
    """
    file_list = [file for file in os.listdir(output_dir) if not file.endswith('2.json')]
    
    for file in file_list:
        labels = []  # 实际上全是 1（因为只选 has_sar == True 的样本）
        preds = []   # 1 表示正确识别对象，0 表示未正确识别
        
        file_path = os.path.join(output_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if not item.get('has_sar', False):
                continue  # 跳过无讽刺样本
            
            # 真实有讽刺：标签为 1
            labels.append(1)
            
            # 判断是否正确识别对象
            # 条件：模型认为有讽刺（llm_has_sar）且 obj_flag 为 True
            if item.get('llm_has_sar', False) and item.get('obj_flag', False):
                preds.append(1)
            else:
                preds.append(0)
        
        if len(labels) == 0:
            print(f"File: {file} - No sarcastic samples found.")
            continue
        
        # 计算指标（由于 label 全为 1，precision = recall = f1 = accuracy）
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        accuracy = accuracy_score(labels, preds)
        
        print("=" * 40)
        print(f"File: {file}")
        print(f"Total sarcastic samples: {len(labels)}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print("=" * 40)

def main():
    """
    主函数，解析命令行参数并执行处理流程
    """
    parser = argparse.ArgumentParser(description='处理讽刺检测结果并计算评估指标')
    parser.add_argument('--input_dir', required=True, help='输入目录路径（包含待处理的JSON文件）')
    parser.add_argument('--output_dir', required=True, help='输出目录路径（处理后的文件存放位置）')
    
    args = parser.parse_args()
    
    # 处理文件
    process_files(args.input_dir, args.output_dir)
    
    # 计算评估指标
    calculate_metrics(args.output_dir)

if __name__ == "__main__":
    main()