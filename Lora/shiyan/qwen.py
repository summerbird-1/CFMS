import argparse
import json
import os
from PIL import Image

# ====== lmdeploy 相关导入 ======
from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN

# 全局 pipeline（将在 main 中初始化）
pipe = None

def ensure_jpg(image_path):
    """
    确保图片为JPEG格式，如果不是则转换（lmdeploy 更兼容 JPG）
    """
    with Image.open(image_path) as img:
        if img.format != 'JPEG':
            base, _ = os.path.splitext(image_path)
            new_path = base + '_converted.jpg'
            img.convert('RGB').save(new_path, 'JPEG')
            return new_path
        else:
            return image_path

def build_prompt(text):
    """
    构建提示词
    """

    prompt = f"""给你一张图片，图片配文为:{text}。分析该图文对是否含有讽刺，并给出讽刺对象和解释，按以下格式回答:是否讽刺：回答是或否 讽刺对象：具体事物,无讽刺则为空 讽刺解释：对于讽刺的简短解释，无讽刺则为空"""
    return prompt

def qwenvl_anno(prompt, image_path):
    """
    使用 lmdeploy pipeline 进行推理
    """
    # lmdeploy.vl 要求输入是 (prompt_with_image_token, [PIL.Image])
    # 注意：prompt 中必须包含 IMAGE_TOKEN（即 '<image>'）
    full_prompt = prompt + "\n" + IMAGE_TOKEN  # 或者你也可以把 IMAGE_TOKEN 插入到 prompt 中合适位置
    image = load_image(image_path)
    response = pipe((full_prompt, [image]))
    return response.text

def process_sarcasm_detection(input_path, output_path):
    """
    处理讽刺检测任务
    """
    with open(input_path, 'r') as f:
        testdata = json.load(f)
    
    qwenvl_res = []
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            qwenvl_res = json.load(f)
    
    processed_images = {item.get('image_path', '') for item in qwenvl_res}
    
    for item in testdata[2:]:
        image_path = item['image_path']
        
        text = item['text']
        prompt = build_prompt(text)
        print(f"Processing: {image_path}")
        print(prompt)

        try:
            # 可选：转换为 JPG（lmdeploy 对非 JPG 支持可能不稳定）
            img_path = ensure_jpg(image_path)
            answer = qwenvl_anno(prompt, img_path)
            item['llm_res'] = answer
            qwenvl_res.append(item)
            
            with open(output_path, 'w') as f:
                json.dump(qwenvl_res, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

def main():
    global pipe
    parser = argparse.ArgumentParser(description='使用Qwen2.5-VL (lmdeploy) 进行讽刺检测')
    parser.add_argument('--input_path', required=True, help='输入数据JSON文件路径')
    parser.add_argument('--output_path', required=True, help='输出结果文件路径')
    parser.add_argument('--model_path', default='/home/zhangjunzhao/zjz/XHSS/fintune/LLaMA-Factory/output/qwen2_5vl_lora_sft', help='模型路径')
    
    args = parser.parse_args()

    # 初始化 lmdeploy pipeline
    print("Loading model with lmdeploy...")
    pipe = pipeline(
        args.model_path,
        backend_config=TurbomindEngineConfig(
            session_len=8192,  # Qwen-VL 通常不需要 16k，8k 足够
            cache_max_entry_count=0.8,
            tp=1  # 如果多卡可调
        )
    )
    print("Model loaded.")

    process_sarcasm_detection(args.input_path, args.output_path)

if __name__ == "__main__":
    main()