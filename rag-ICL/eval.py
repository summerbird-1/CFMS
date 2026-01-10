import re
import json
import os
import argparse
def extract_satire_info1(text):
    text = text.replace('\n', '')
    pattern = r'是否讽刺:(?P<是否讽刺>.*?);讽刺对象:(?P<讽刺对象>.*?);讽刺解释:(?P<讽刺解释>.*)'
    match = re.search(pattern, text)
    if match:
        result = match.groupdict()
        return result

def process_json_files(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filelist = [file for file in os.listdir(input_dir) if file.endswith('.json')]
    print(filelist)
    for filename in filelist:
        print(filename)
        filepath = os.path.join(input_dir,filename)
        output_path = os.path.join(output_dir,filename)
        with open(filepath, 'r') as f:
            annodata = json.load(f)
            for item in annodata:
                llm_res = item['llm_res']
                
                result = extract_satire_info1(llm_res)
                if result is None:
                    print(llm_res)
                try:
                    has_sar_llm = result['是否讽刺']
                    sar_obj_llm = result['讽刺对象']
                    sar_exp_llm = result['讽刺解释']
                except:
                    print(f"文件{filename}中样本数据格式不正确{item},请手动纠正")
                if has_sar_llm.strip() == '否':
                    item['has_sar_llm'] = False
                    item['sar_obj_llm'] = ''
                    item['sar_exp_llm'] = ''
                else:
                    item['has_sar_llm'] = True
                    item['sar_obj_llm'] = sar_obj_llm
                    item['sar_exp_llm'] = sar_exp_llm
        with open(output_path, 'w') as f:
            json.dump(annodata, f, ensure_ascii=False, indent=4)
            

def calculate_metrics(input_dir):
    # input_dir = './rag_processed'
    filelist = [file for file in os.listdir(input_dir) if file.endswith('.json')]
    for filename in filelist:
        print(filename)
        filepath = os.path.join(input_dir,filename)
        with open(filepath, 'r') as f:
            annodata = json.load(f)
            TP, TN, FP, FN = 0, 0, 0, 0
            for item in annodata:
                has_sar = item['has_sar']
                has_sar_llm = item['has_sar_llm']
                if has_sar and has_sar_llm:
                    TP += 1
                elif not has_sar and not has_sar_llm:
                    TN += 1
                elif not has_sar and has_sar_llm:
                    FP += 1
                elif has_sar and not has_sar_llm:
                    FN += 1
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
            print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}, Accuracy: {accuracy:.4f}')
if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='评估')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    # 解析命令行参数
    args = parser.parse_args()
    process_json_files(args.input_dir, args.output_dir)
    calculate_metrics(args.output_dir)