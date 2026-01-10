from lmdeploy import pipeline, TurbomindEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.constants import IMAGE_TOKEN
import json
import os
import argparse
import time
model = '/home/fintune/LLaMA-Factory/OpenGVLab/InternVL2_5-8B-hf'
pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=16384))

def intervl_anno(testpath,example_path,res_path):
    with open(testpath,'r') as f:
        testdata = json.load(f)
    with open(example_path,'r') as f:
        examples = json.load(f)
    intervl_res = json.load(open(res_path)) if os.path.exists(res_path) else []
    for i in range(len(testdata)):
        image_urls = []
        mate_prompt = "Example{}:\n 输入：图片配文为:{};图片为: {}\n 输出：是否讽刺:{};讽刺对象:{};讽刺解释:{}\n"
        user_prompt = "给你一张图片,分析该图片是否含有讽刺，并给出讽刺对象和解释。\n{} Test:\n输入：图片配文为:{};图片为: {}\n输出："
        exam_prompts = ""
        for j in range(len(examples[i])):
            ex = examples[i][j]
            if ex['has_sar']:
                has_sar = '是'
            else:
                has_sar = '否'
            if type(ex['sar_obj']) == list:
                sar_obj = ex['sar_obj'][0] if ex['has_sar'] else '无'
                sar_exp = ex['sar_exp'][0] if ex['has_sar'] else '无'
            else:
                sar_obj = ex['sar_obj'] if ex['has_sar'] else '无'
                sar_exp = ex['sar_exp'] if ex['has_sar'] else '无'
            example_prompt = mate_prompt.format(
                j+1, ex['text'], IMAGE_TOKEN,has_sar,
                sar_obj,
                sar_exp
            )
            exam_prompts += example_prompt
            image_urls.append(ex['image_path'])


        text = testdata[i]['text']
        image_urls.append(testdata[i]['image_path'])
        prompt =user_prompt.format(
            exam_prompts, text, IMAGE_TOKEN
        )
        print(prompt)
        images = [load_image(img_url) for img_url in image_urls]
        response = pipe((prompt, images))
        print(response.text)
        testdata[i]['llm_res'] = response.text
        intervl_res.append(testdata[i])
    with open(res_path,'w') as f:
        json.dump(intervl_res, f, ensure_ascii=False, indent=4)
        
        
if __name__ == '__main__':
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='使用InternVL进行讽刺检测')
    
    # 添加参数
    parser.add_argument('--model', type=str, default='/home/LLMs/InternVL/InternVL2_5-8B',
                        help='模型路径')
    parser.add_argument('--testpath', type=str, required=True,
                        help='测试数据路径')
    parser.add_argument('--example_path', type=str, required=True,
                        help='示例数据路径')
    parser.add_argument('--res_path', type=str, default='intervl_anno_rag_shot.json',
                        help='结果保存路径')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用主函数
    intervl_anno(args.testpath, args.example_path, args.res_path)