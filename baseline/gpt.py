import argparse
import json
import os
import base64
from openai import OpenAI

# 初始化OpenAI客户端
client = OpenAI(
    base_url='',
    api_key='',
)

def encode_image(image_path):
    """
    将图片编码为base64格式
    
    Args:
        image_path (str): 图片路径
        
    Returns:
        str: base64编码的图片
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt_4o(img_path, prompt):
    """
    调用GPT-4o模型进行推理
    
    Args:
        img_path (str): 图片路径
        prompt (str): 提示词
        
    Returns:
        str: 模型返回结果
    """
    base64_image = encode_image(img_path)
    print("+++" * 20)
    print(f"调用GPT-4o生成文本的请求为：\n{prompt}")
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        }]
    )
    print("+++" * 20)
    return response.choices[0].message.content

def build_prompt(text):
    """
    根据文本内容构建提示词
    
    Args:
        text (str): 图片配文
        
    Returns:
        str: 构建的提示词
    """
    if text == "":
        prompt = """给你一张图片,分析该图片是否含有讽刺，并给出讽刺对象和解释，按以下格式回答:
    <是否讽刺>是或否</是否讽刺>
    <讽刺对象>具体事物</讽刺对象>
    <讽刺解释>对于讽刺的简短解释</讽刺解释>"""
    else:
        prompt = f"""给你一张图片，图片配文为:{text},分析该数据是否含有讽刺，并给出讽刺对象和解释，按以下格式回答:
    <是否讽刺>是或否</是否讽刺>
    <讽刺对象>具体事物</讽刺对象>
    <讽刺解释>对于讽刺的简短解释</讽刺解释>"""
    return prompt

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
    gpt_4o_res = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            gpt_4o_res = json.load(f)
    
    # 创建已处理图片的集合，避免重复处理
    processed_images = {item.get('image_path', '') for item in gpt_4o_res}
    
    # 遍历测试数据
    for item in testdata[2:]:
        image_path = item['image_path']
        
        # 跳过已处理的图片
        if image_path in processed_images:
            continue
            
        text = item['text']
        prompt = build_prompt(text)
        
        try:
            res = gpt_4o(image_path, prompt)
            print(res)
            item['llm_res'] = res
            gpt_4o_res.append(item)
            
            # 保存结果到文件
            with open(output_path, 'w') as f:
                json.dump(gpt_4o_res, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")

def main():
    """
    主函数，解析命令行参数并执行处理流程
    """
    parser = argparse.ArgumentParser(description='使用GPT-4o进行讽刺检测')
    parser.add_argument('--input_path', required=True, help='输入数据JSON文件路径')
    parser.add_argument('--output_path', required=True, help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 执行处理流程
    process_sarcasm_detection(args.input_path, args.output_path)

if __name__ == "__main__":
    main()