import unsloth

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from src.utils.free_gpu import free_gpu
from src.prompt.prompt_utils import build_user_msg_train
from src.utils.constants import FEWSHOT_PATH
import os
import json



def inference_vllm(args, temp=None, force_out_csv=None):
    """Inference với model đã train hoặc base model"""
    print("==> Inference...")
    free_gpu()
    
    with open(FEWSHOT_PATH, "r") as f:
        fewshot_data = json.load(f)
    
    
    sampling_params = SamplingParams(
        temperature=0.1 if temp is None else temp,
        top_p=0.8,
        top_k=5,
        max_tokens=args.max_new_tokens,
    )

    if args.inference_model_path:
        model_path = args.inference_model_path
    else:
        # Kiểm tra model đã train hay chưa
        vllm_model_path = f"{args.out_dir}_vllm"
        if os.path.exists(vllm_model_path):
            print(f"==> Using fine-tuned model: {vllm_model_path}")
            model_path = vllm_model_path
        else:
            print(f"==> Fine-tuned model not found. Using base model: {args.model_name}")
            model_path = args.model_name


    model = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=args.max_seq_len,
        gpu_memory_utilization=0.75,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Load data
    df = pd.read_csv(args.test_csv)
    if bool(args.mock):  # Convert to boolean for clarity
        df = df.head(10)

    required = ["id", "context", "prompt", "response"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột {col} trong test.csv")

    preds, raw_outputs = [], []

    for start in tqdm(range(0, len(df), args.batch_size), desc="Inference"):
        batch = df.iloc[start : start + args.batch_size]

        prompts = []
        for _, row in batch.iterrows():
            user_msg = build_user_msg_train(
                row["context"],
                row["prompt"],
                row["response"],
                fewshot_data[:5],
            )
            conversation = [{"role": "user", "content": user_msg}]
            input_text = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompts.append(input_text)

        # Generate song song với vLLM
        outputs = model.generate(prompts, sampling_params)

        for out in outputs:
            decoded = out.outputs[0].text
            raw_outputs.append(decoded)

            # Parse nhãn
            label = "no_label"
            for lab in ["extrinsic", "intrinsic", "no"]:
                if lab in decoded.lower():
                    label = lab
                    break
            preds.append(label)

    df["predict_label"] = preds
    df["raw_output"] = raw_outputs
    df = df[["id", "predict_label"]]
    
    out_csv = force_out_csv if force_out_csv else args.out_csv
    
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"==> Inference done with vLLM! Saved to {out_csv}")
