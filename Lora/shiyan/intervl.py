import argparse
import json
import os
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image

# 初始化模型
# model = '/home/LLMs/InternVL/InternVL2_5-8B'
model = '/home/zhangjunzhao/zjz/XHSS/fintune/LLaMA-Factory/output/internlora_sft'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=8192))

def build_prompt(text):
    """
    根据文本内容构建提示词
    
    Args:
        text (str): 图片配文
        
    Returns:
        str: 构建的提示词
    """
    prompt = f"""给你一张图片，图片配文为:{text}。分析该图文对是否含有讽刺，并给出讽刺对象和解释，按以下格式回答:是否讽刺：回答是或否 讽刺对象：具体事物,无讽刺则为空 讽刺解释：对于讽刺的简短解释，无讽刺则为空"""
    print(prompt)
    return prompt

def get_image_path(original_path):
    """
    获取有效的图片路径，如果原始路径不存在则尝试备份路径
    
    Args:
        original_path (str): 原始图片路径
        
    Returns:
        str: 有效的图片路径
    """
    if os.path.isfile(original_path):
        return original_path
    else:
        # 尝试替换为备份路径
        backup_path = original_path.replace('sorted_images',
                                          'sorted_images/duplicates_backup', 1)
        if os.path.isfile(backup_path):
            return backup_path
        else:
            raise FileNotFoundError(f"图片文件未找到: {original_path} 或 {backup_path}")

def intervl_anno(prompt, img_path):
    """
    调用InternVL模型进行推理
    
    Args:
        prompt (str): 提示词
        img_path (str): 图片路径
        
    Returns:
        str: 模型返回结果
    """
    image = load_image(img_path)
    answer = pipe((prompt, image)).text
    print(answer)
    
    # 如果答案是'是'，重新生成（这部分逻辑保持原样）
    while answer == '是':
        answer = pipe((prompt, image)).text
        
    return answer

def process_sarcasm_detection(input_path, output_path):
    """
    处理讽刺检测任务
    
    Args:
        input_path (str): 输入数据路径
        output_path (str): 输出结果路径
    """
    # 加载测试数据
    with open(input_path, 'r') as f:
        testdata = json.load(f)
    
    # 如果输出文件已存在，则加载已有结果
    intervl_res = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            intervl_res = json.load(f)
    
    # 创建已处理图片的集合，避免重复处理
    processed_images = {item.get('image_path', '') for item in intervl_res}
    
    # 遍历测试数据
    for item in testdata[2:]:
        image_path = item['image_path']
        

        text = item['text']
        prompt = build_prompt(text)
        
        try:
            # 获取有效图片路径
            img_path = get_image_path(image_path)
            
            # 调用模型获取结果
            answer = intervl_anno(prompt, img_path)
            
            item['llm_res'] = answer
            intervl_res.append(item)
            
            # 保存结果到文件
            with open(output_path, 'w') as f:
                json.dump(intervl_res, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")

def main():
    """
    主函数，解析命令行参数并执行处理流程
    """
    parser = argparse.ArgumentParser(description='使用InternVL进行讽刺检测')
    parser.add_argument('--input_path', required=True, help='输入数据JSON文件路径')
    parser.add_argument('--output_path', required=True, help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 执行处理流程
    process_sarcasm_detection(args.input_path, args.output_path)

if __name__ == "__main__":
    main()