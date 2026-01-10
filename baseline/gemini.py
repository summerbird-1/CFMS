import argparse
import json
import os
from google import genai
from google.genai import types

# 初始化Gemini客户端（保持原有配置）
client = genai.Client(
    api_key="",
    vertexai=True,  # 优先使用vertexai协议访问，稳定性更高
    http_options={
        "base_url": ""
    },
)

def gemini(img_path, prompt):
    with open(img_path, 'rb') as f:
        image_bytes = f.read()
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
            ),
            prompt
        ]
    )
    print("+++" * 20)
    return response.text

def build_prompt(text):
    """
    根据文本构建提示词
    
    Args:
        text (str): 图片配文
        
    Returns:
        str: 构建好的提示词
    """
    if text == "":
        prompt = """给你一张图片,分析该图片是否含有讽刺，并给出讽刺对象和解释，按以下格式回答:
    <是否讽刺>回答是或否</是否讽刺>
    <讽刺对象>具体事物</讽刺对象>
    <讽刺解释>对于讽刺的简短解释</讽刺解释>"""
    else:
        prompt = f"""给你一张图片，图片配文为:{text},分析该数据是否含有讽刺，并给出讽刺对象和解释，按以下格式回答:
    <是否讽刺>回答是或否</是否讽刺>
    <讽刺对象>具体事物</讽刺对象>
    <讽刺解释>对于讽刺的简短解释</讽刺解释>"""
    return prompt

def process_sarcasm_detection(testdata_path, output_path):
    """
    处理讽刺检测任务
    
    Args:
        testdata_path (str): 测试数据路径
        output_path (str): 输出结果路径
    """
    # 加载测试数据
    with open(testdata_path, 'r') as f:
        testdata = json.load(f)
    
    # 如果输出文件已存在，则加载已有结果
    gemini_res = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            gemini_res = json.load(f)
    
    # 创建已处理图片的集合，避免重复处理
    processed_images = {item.get('image_path', '') for item in gemini_res}
    
    # 遍历测试数据
    for item in testdata[486:]:
        image_path = item['image_path']
        
        # 跳过已处理的图片
        if image_path in processed_images:
            continue
            
        text = item['text']
        prompt = build_prompt(text)
        
        try:
            res = gemini(image_path, prompt)
            print(res)
            item['llm_res'] = res
            gemini_res.append(item)
            
            # 保存结果到文件
            with open(output_path, 'w') as f:
                json.dump(gemini_res, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"处理图片 {image_path} 时发生错误: {e}")

def main():
    """
    主函数，解析命令行参数并执行处理流程
    """
    parser = argparse.ArgumentParser(description='使用Gemini进行讽刺检测')
    parser.add_argument('--testdata_path', required=True, help='测试数据JSON文件路径')
    parser.add_argument('--output_path', required=True, help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 执行处理流程
    process_sarcasm_detection(args.testdata_path, args.output_path)

if __name__ == "__main__":
    main()