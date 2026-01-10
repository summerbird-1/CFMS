mkdir -p infer_res
#gemini跑baselinezh
python gemini.py --testdata_path ../cfmsdata/testdata.json --output_path ./infer_res/gemini_anno.json

#gpt跑baselinezh
 python gpt.py --input_path ../cfmsdata/testdata.json --output_path ./infer_res/gpt_anno.json

#qwenvl跑baselinezh
python Qwen2.5-vl.py --input_path ../cfmsdata/testdata.json --output_path ./infer_res/qwen_anno.json

#internvl跑baselinezh
python intervl.py --input_path ../cfmsdata/testdata.json --output_path ./infer_res/intervl_anno.json


python llava.py ../cfmsdata/testdata.json ./infer_res/llava_anno.json

python eval.py --input_dir ./infer_res --output_dir ./infer_res_class
python eval_obj.py --input_dir ./infer_res_class --output_dir ./infer_res_obj
python eval_exp_bert.py --input_dir ./infer_res 