echo "Building RAG 1-shot examples..."
python build_rag.py \
--traindata ../cfmsdata/traindata_desc.json \
--testdata ../cfmsdata/testdata_desc.json \
--output retrieved_examples_withdesc_1shot.json \
--k_shots 1

mkdir -p rag_res

echo "Qwen2.5-vl + RAG 1-shot test"
python Qwen2.5-vl.py --testpath ../cfmsdata/testdata.json --example_path retrieved_examples_1shot.json --res_path ./rag_res/qwenvl_anno_rag_1shot.json

echo "Intervl + RAG 1-shot test"
python intervl.py --testpath ../cfmsdata/testdata.json --example_path retrieved_examples_1shot.json --res_path ./rag_res/intervl_anno_rag_1shot.json

echo "Qwen2.5-vl + RAG 1-shot test"
python Qwen2.5-vl.py --testpath ../cfmsdata/testdata.json --example_path retrieved_examples_withdesc_1shot.json --res_path ./rag_res/qwenvl_anno_withdesc_rag_1shot.json

echo "Intervl + RAG 1-shot test"
python intervl.py --testpath ../cfmsdata/testdata.json --example_path retrieved_examples_withdesc_1shot.json --res_path ./rag_res/intervl_anno_withdesc_rag_1shot.json


python gemini.py --testpath ./testdata.json --example_path retrieved_examples_1shot.json --res_path ./rag_res/gemini_anno_rag_1shot.json

python gpt.py --testpath ./testdata.json --example_path random_1_shot.json --res_path ./rag_res/gpt_anno_random_1shot.json
python gemini.py --testpath ./testdata.json --example_path random_1_shot.json --res_path ./rag_res/gemini_anno_random_1shot.json

echo "Qwen2.5-vl + random 1-shot test"
python Qwen2.5-vl.py --testpath ../cfmsdata/testdata.json --example_path random_1_shot.json --res_path ./rag_res/qwenvl_anno_random_1shot.json

echo "Intervl + random 1-shot test"
python intervl.py --testpath ../cfmsdata/testdata.json --example_path random_1_shot.json --res_path ./rag_res/intervl_anno_random_1shot.json

echo "Processing results..."
python eval.py --input_dir './rag_res' --output_dir './rag_class'

nohup python eval_obj.py --input_dir ./rag_class --output_dir ./rag_obj > log &

python eval_exp.py --input_dir  ./rag_obj