if [ ! -d "uit_dsc_lora_model_final_vllm_submit" ]; then
	echo "Downloading VLLM..."
	git lfs install
    git clone https://huggingface.co/thangquang09/uit_dsc_lora_model_final_vllm_submit
fi


pip install -r requirements.txt

python3 main.py \
	--mode inference \
	--test_csv data/test/vihallu-private-test.csv \
	--inference_model_path uit_dsc_lora_model_final_vllm_submit \
	--model_name unsloth/Qwen3-4B-Instruct-2507 \
	--max_seq_len 5000 \
	--max_new_tokens 64 \
	--batch_size 8 \
	--out_csv submit.csv \
	--mock 0