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
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def gpt_4o_fewshot(image_paths, prompt):
    """
    调用GPT-4o模型进行few-shot推理，支持多张图片
    """
    content = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        base64_image = encode_image(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })
    
    print("+++" * 20)
    print(f"调用GPT-4o生成文本的请求为：\n{prompt}")
    response = client.chat.completions.create(
        model="gpt-4o-2024-11-20",
        messages=[{"role": "user", "content": content}]
    )
    print("+++" * 20)
    return response.choices[0].message.content

def build_fewshot_prompt(test_text, examples):
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

def gpt4o_fewshot_anno(testpath, example_path, res_path):
    # 加载数据
    with open(testpath, 'r') as f:
        testdata = json.load(f)
    with open(example_path, 'r') as f:
        examples_list = json.load(f)

    intervl_res = []
    if os.path.exists(res_path):
        with open(res_path, 'r') as f:
            intervl_res = json.load(f)
    
    processed_ids = {item['image_path'] for item in intervl_res}

    for i in range(200,len(testdata)):
        test_item = testdata[i]
        if test_item['image_path'] in processed_ids:
            continue

        examples = examples_list[i]  # 每个测试样本对应一组 examples
        test_text = test_item['text']

        # 构建prompt
        prompt= build_fewshot_prompt(test_text, examples)

        # 收集所有图片路径：examples + test
        image_paths = [ex['image_path'] for ex in examples] + [test_item['image_path']]

        try:
            response_text = gpt_4o_fewshot(image_paths, prompt)
            print(f"第 {i} 个样本的 LLM 输出为：\n{response_text}")
            test_item['llm_res'] = response_text
            intervl_res.append(test_item)

            # 实时保存
            with open(res_path, 'w') as f:
                json.dump(intervl_res, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"处理第 {i} 个样本时出错: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='GPT-4o Few-shot 讽刺检测')
    parser.add_argument('--testpath', type=str, required=True, help='测试数据路径')
    parser.add_argument('--example_path', type=str, required=True, help='示例数据路径（每个测试样本对应一组examples）')
    parser.add_argument('--res_path', type=str, required=True, help='结果保存路径')
    args = parser.parse_args()

    gpt4o_fewshot_anno(args.testpath, args.example_path, args.res_path)

if __name__ == "__main__":
    main()
