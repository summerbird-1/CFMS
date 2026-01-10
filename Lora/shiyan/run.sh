mkdir -p infer_res

CUDA_VISIBLE_DEVICES=3 python llava.py  ../cfmsdata/testdata.json  ./infer_res/llava_anno.json
CUDA_VISIBLE_DEVICES=3 python qwen.py --input_path ../cfmsdata/testdata.json --output_path ./infer_res/qwen_anno.json
CUDA_VISIBLE_DEVICES=3 python intervl.py --input_path ../cfmsdata/testdata.json --output_path ./infer_res/intervl_anno.json
python eval.py --input_dir ./infer_res --output_dir ./infer_res_class
python eval_obj.py --input_dir ./infer_res_class --output_dir ./infer_res_obj
python eval_exp_bert.py --input_dir ./infer_res