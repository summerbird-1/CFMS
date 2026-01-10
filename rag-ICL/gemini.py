import argparse
import json
import os
import time
from google import genai
from google.genai import types

# 初始化Gemini客户端
client = genai.Client(
    api_key="",
    vertexai=True,
    http_options={
        "base_url": ""
    },
)

def gemini_with_examples(image_paths, prompt):
    """
    使用Gemini处理多张图片和文本提示
    
    Args:
        image_paths (list): 图片路径列表
        prompt (str): 文本提示
        
    Returns:
        str: 模型响应
    """
    # 准备多模态内容
    parts = []
    
    # 添加所有图片
    for img_path in image_paths:
        with open(img_path, 'rb') as f:
            image_bytes = f.read()
        parts.append(types.Part.from_bytes(
            data=image_bytes,
            mime_type='image/jpeg',
        ))
    
    # 添加文本提示
    parts.append(prompt)
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=parts
    )
    return response.text

def build_fewshot_prompt(test_text, examples):
    """
    构建few-shot提示词（参考您提供的格式）
    
    Args:
        test_text (str): 测试样本的文本
        examples (list): 示例数据列表
        
    Returns:
        str: 构建好的提示词
    """
    mate_prompt = "示例{}：\n输入：图片配文为“{}”；图片为[IMAGE{}]。\n输出：是否讽刺:{}；讽刺对象:{}；讽刺解释:{}\n\n"
    exam_prompts = ""
    
    for j, ex in enumerate(examples):
        has_sar = '是' if ex['has_sar'] else '否'
        sar_obj = ex['sar_obj'][0] if ex['has_sar'] and isinstance(ex['sar_obj'], list) else (ex['sar_obj'] if ex['has_sar'] else '无')
        sar_exp = ex['sar_exp'][0] if ex['has_sar'] and isinstance(ex['sar_exp'], list) else (ex['sar_exp'] if ex['has_sar'] else '无')
        exam_prompts += mate_prompt.format(
            j + 1, ex['text'], j + 1, has_sar, sar_obj, sar_exp
        )

    # 明确指示只回答 Test，并给出干净格式
    test_img_idx = len(examples) + 1
    prompt = (
        f"{exam_prompts}"
        f"测试样本：\n输入：图片配文为“{test_text}”；图片为[IMAGE{test_img_idx}]。\n\n"
        "请仅分析上述测试样本，并严格按以下格式输出，不要包含任何其他文字、标题、示例或解释：\n"
        "是否讽刺: 是或否\n"
        "讽刺对象: ...\n"
        "讽刺解释: ...\n"
    )
    return prompt

def process_sarcasm_detection(testdata_path, example_path, output_path):
    """
    处理讽刺检测任务（few-shot版本）
    
    Args:
        testdata_path (str): 测试数据路径
        example_path (str): 示例数据路径
        output_path (str): 输出结果路径
    """
    # 加载测试数据和示例数据
    with open(testdata_path, 'r', encoding='utf-8') as f:
        testdata = json.load(f)
    
    with open(example_path, 'r', encoding='utf-8') as f:
        examples_data = json.load(f)
    
    # 如果输出文件已存在，则加载已有结果
    gemini_res = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            gemini_res = json.load(f)
    
    # 创建已处理样本的集合，避免重复处理
    processed_indices = {item.get('index', i) for i, item in enumerate(gemini_res)}
    
    # 遍历测试数据
    for i, item in enumerate(testdata):
        # 跳过已处理的样本
        if i in processed_indices:
            continue
        
        print(f"Processing sample {i+1}/{len(testdata)}")
        
        # 获取当前样本对应的示例
        examples = examples_data[i] if i < len(examples_data) else []
        
        # 构建提示词
        prompt = build_fewshot_prompt(item['text'], examples)
        print("=== Prompt ===")
        print(prompt)
        print("=============")
        
        # 准备所有图片路径（示例图片 + 测试图片）
        image_paths = []
        for ex in examples:
            if 'image_path' in ex:
                image_paths.append(ex['image_path'])
        
        # 添加测试图片
        image_paths.append(item['image_path'])
        
        try:
            # 调用Gemini API
            res = gemini_with_examples(image_paths, prompt)
            print("+++" * 20)
            print(f"Response for sample {i}:")
            print(res)
            
            # 保存结果
            item['llm_res'] = res
            item['index'] = i  # 保存索引便于去重
            gemini_res.append(item)
            
            # 定期保存结果，防止中断丢失
            if len(gemini_res) % 5 == 0:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(gemini_res, f, ensure_ascii=False, indent=4)
                print(f"Saved {len(gemini_res)} results to {output_path}")
            
            # 添加延迟避免API限制
            time.sleep(1)
            
        except Exception as e:
            print(f"处理样本 {i} 时发生错误: {e}")
            # 保存已处理的结果
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(gemini_res, f, ensure_ascii=False, indent=4)
            continue
    
    # 最终保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(gemini_res, f, ensure_ascii=False, indent=4)
    print(f"处理完成！总共处理了 {len(gemini_res)} 个样本")
def main():
    """
    主函数，解析命令行参数并执行处理流程
    """
    parser = argparse.ArgumentParser(description='使用Gemini进行讽刺检测（Few-shot版本）')
    parser.add_argument('--testpath', required=True, help='测试数据JSON文件路径')
    parser.add_argument('--example_path', required=True, help='示例数据JSON文件路径')
    parser.add_argument('--res_path', required=True, help='输出结果文件路径')
    
    args = parser.parse_args()
    
    # 执行处理流程
    process_sarcasm_detection(args.testpath, args.example_path, args.res_path)

if __name__ == "__main__":
    import time
    main()
