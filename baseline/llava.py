import argparse
import requests
from PIL import Image
import json
import torch
from modelscope import AutoProcessor, LlavaForConditionalGeneration
import os

def llava_anno(prompt, image_path, model, processor):
    """
    使用LLaVA模型对图像和文本进行分析
    
    Args:
        prompt (str): 输入的提示文本
        image_path (str): 图像文件路径
        model: 预训练的LLaVA模型
        processor: 模型处理器
    
    Returns:
        str: 模型的回答结果
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt}
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    
    # 将张量移动到GPU
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=512)
    
    # 解码结果并提取回答部分
    full_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    answer = full_text.split("ASSISTANT:")[-1].strip()
    print(answer)
    return answer

def main(input_file, output_file):
    """
    主函数，处理输入数据并生成结果
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
    """
    # 默认模型路径
    model_id = "/home/LLMs/llava-hf/llava-1.5-7b-hf"
    
    # 加载测试数据
    with open(input_file, 'r') as f:
        testdata = json.load(f)
    
    # 如果输出文件已存在，则加载已有结果
    llava_res = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            llava_res = json.load(f)
        processed_count = len(llava_res)
        print(f"已存在 {processed_count} 条结果，从第 {processed_count} 条开始继续处理...")
        start_index = processed_count
    else:
        start_index = 0
    
    # 加载模型和处理器
    print("正在加载模型...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(0)
    
    processor = AutoProcessor.from_pretrained(model_id)
    print("模型加载完成")
    
    # 处理数据
    for i, item in enumerate(testdata[start_index:], start=start_index):
        text = item['text']
        prompt = f"""给你一张图片，图片配文为:{text}。分析该图文对是否含有讽刺，并给出讽刺对象和解释，按以下格式回答:是否讽刺：回答是或否; 讽刺对象：具体事物,无讽刺则为空; 讽刺解释：对于讽刺的简短解释，无讽刺则为空"""
        print(prompt)
        # 处理图像路径
        img_path = item['image_path']
       
        # 获取模型预测结果
        answer = llava_anno(prompt, img_path, model, processor)
        
        # 如果答案是"是"，重新获取结果（原代码逻辑）
        while answer == '是':
            answer = llava_anno(prompt, img_path, model, processor)
            
        item['llm_res'] = answer
        llava_res.append(item)
        
        # 实时保存结果
        with open(output_file, 'w') as f:
            json.dump(llava_res, f, ensure_ascii=False, indent=4)
        
        print(f"已完成第 {i+1}/{len(testdata)} 条数据处理")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用LLaVA模型处理图像讽刺检测任务")
    parser.add_argument("input_file", type=str, help="输入JSON文件路径")
    parser.add_argument("output_file", type=str, help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    main(args.input_file, args.output_file)