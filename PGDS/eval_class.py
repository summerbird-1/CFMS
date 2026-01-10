import argparse
import json
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def parse_satire_response(text):

    # 允许中文/英文冒号、分号，前后可有空格，兼容换行
    pattern = r"是否讽刺[：:]\s*([^\n；;]+?)[；;]\s*讽刺对象[：:]\s*([^\n；;]*?)[；;]\s*讽刺解释[：:]\s*(.*?)(?=\s*$)"
    
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return {
            "is_satirical": match.group(1).strip(),
            "satirical_target": match.group(2).strip(),
            "satirical_explanation": match.group(3).strip()
        }
    else:
        print("无法解析结果")
        print(text)

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
    filelist = [file for file in os.listdir(input_dir) if file.endswith('.json')]
    
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
                llm_res = item['llm_res']
                res = parse_satire_response(llm_res)
                
                # 转换结果格式
                if '是' in res['is_satirical']:
                    item['llm_has_sar'] = True
                else :
                    item['llm_has_sar'] = False
                    
                item['llm_sar_obj'] = [res['satirical_target']]
                item['llm_sar_exp'] = [res['satirical_explanation']]
            except Exception as e:
                print(f"{item},该条数据无法提取 sarcasm info: {e}")
        
        # 保存处理后的数据
        with open(output_path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"{file} processed")

def calculate_metrics(output_dir):
    """
    计算二分类指标
    
    Args:
        output_dir (str): 处理后文件的目录路径
    """
    # 获取所有JSON文件
    file_list = [file for file in os.listdir(output_dir) if file.endswith('.json')]
    
    for file in file_list:
        label = []
        pred = []
        file_path = os.path.join(output_dir, file)
        
        # 加载数据
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 收集标签和预测结果
        for item in data:
            if item['has_sar']:
                label.append(1)
            else:
                label.append(0)
                
            if item['llm_has_sar']:
                pred.append(1)
            else:
                pred.append(0)
        
        # 计算并打印指标
        print("="*30)
        print(file)
        print(f"Accuracy: {accuracy_score(label, pred)}")
        print(f"Precision: {precision_score(label, pred)}")
        print(f"Recall: {recall_score(label, pred)}")
        print(f"F1-score: {f1_score(label, pred)}")
        print("="*30)

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