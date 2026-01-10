import argparse
import json
import os
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def parse_satire_response(text):
    """
    解析讽刺检测的响应结果
    
    Args:
        text (str): 模型返回的文本结果
        
    Returns:
        dict: 解析后的结果字典
    """
    patterns = {
    # 核心：(?:</标签名>|<|$) → 匹配闭合标签 / 下一个标签开头 / 文本末尾（解决缺失闭合）
    "是否讽刺": r'<是否讽刺>\s*([\s\S]*?)(?:</是否讽刺>|<|$)',
    "讽刺对象": r'<讽刺对象>\s*([\s\S]*?)(?:</讽刺对象>|<|$)',
    "讽刺解释": r'<讽刺解释>\s*([\s\S]*?)(?:</讽刺解释>|<|$)'
}

    def extract_satire_content(text):

        result = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                # 清理内容首尾空白（包括换行/空格）
                content = match.group(1).strip()
                result[key] = content if content else ""  # 空内容返回""，而非空字符串
            else:
                result[key] = None  # 标签完全缺失返回None（可改为""）
        return result

    result = extract_satire_content(text)
    
    return {
        "is_satirical": result["是否讽刺"],
        "satirical_target": result["讽刺对象"],
        "satirical_explanation": result["讽刺解释"]
    }

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
                print(f"{file}出错")
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
    file_list = [file for file in os.listdir(output_dir) if not file.endswith('2.json')]
    
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