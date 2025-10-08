import unsloth

import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.utils.free_gpu import free_gpu
from src.prompt.prompt_utils import build_user_msg_train
from src.utils.constants import FEWSHOT_PATH
import os
import json
import tempfile
import shutil


def merge_lora_with_base_model(base_model_name, lora_path, output_dir=None):
    """
    Merge LoRA adapter với base model để sử dụng với vLLM
    
    Args:
        base_model_name: Tên hoặc path của base model
        lora_path: Path đến LoRA adapter
        output_dir: Thư mục để lưu merged model (nếu None sẽ tạo temp dir)
    
    Returns:
        Path đến merged model
    """
    try:
        from peft import PeftModel
    except ImportError:
        raise ImportError("peft library is required for LoRA support. Install with: pip install peft")
    
    print(f"==> Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"==> Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    print("==> Merging LoRA with base model...")
    merged_model = model.merge_and_unload()
    
    # Tạo output directory nếu không được cung cấp
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="merged_model_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"==> Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir)
    
    # Copy tokenizer từ base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)
    
    print("==> LoRA merge completed successfully!")
    return output_dir



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

    # Xử lý model path
    merged_model_dir = None  # Track temporary merged model directory
    
    if args.use_lora:
        # LoRA mode: merge LoRA adapter với base model
        if not args.inference_model_path:
            raise ValueError("--inference_model_path is required when using --use_lora")
        if not args.model_name:
            raise ValueError("--model_name (base model) is required when using --use_lora")
        
        print("==> LoRA mode enabled")
        merged_model_dir = merge_lora_with_base_model(
            base_model_name=args.model_name,
            lora_path=args.inference_model_path
        )
        model_path = merged_model_dir
        tokenizer_path = merged_model_dir
        
    elif args.inference_model_path:
        # Direct model path (VLLM format)
        model_path = args.inference_model_path
        tokenizer_path = args.inference_model_path
    else:
        # Kiểm tra model đã train hay chưa
        vllm_model_path = f"{args.out_dir}_vllm"
        if os.path.exists(vllm_model_path):
            print(f"==> Using fine-tuned model: {vllm_model_path}")
            model_path = vllm_model_path
            tokenizer_path = vllm_model_path
        else:
            print(f"==> Fine-tuned model not found. Using base model: {args.model_name}")
            model_path = args.model_name
            tokenizer_path = args.model_name

    try:
        model = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=args.max_seq_len,
            gpu_memory_utilization=0.75,
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
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
        
    except Exception as e:
        print(f"Error during inference: {e}")
        raise
    finally:
        # Cleanup temporary merged model directory if created
        if merged_model_dir and os.path.exists(merged_model_dir):
            print(f"==> Cleaning up temporary merged model: {merged_model_dir}")
            shutil.rmtree(merged_model_dir, ignore_errors=True)
