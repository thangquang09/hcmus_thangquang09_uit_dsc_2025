import unsloth

import argparse
import shutil
import time
import os
from src.task.train import train
from src.task.inference import inference_vllm


def parse_args():
    p = argparse.ArgumentParser()
    # Mode config
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "continue_train", "inference", "inference_random", "inference_1_temp"],
    )
    p.add_argument("--inference_model_path", type=str)
    p.add_argument("--mock", type=int, default=0)
    p.add_argument("--skip_inference", type=int, default=0)

    # Training Config
    p.add_argument("--train_csv", type=str)
    p.add_argument("--test_csv", type=str)
    p.add_argument("--train_from", type=str, help="Path or HuggingFace repo to continue training from")
    p.add_argument("--out_dir", type=str, default="lora_model")
    p.add_argument("--repo_id", type=str, default="thang09/uit_qwen3_thinking")
    p.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Instruct-2507")
    p.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for accessing private models")
    p.add_argument(
        "--max_seq_len", type=int, default=5000
    )
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=3407)

    # Inference Config
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--use_lora", action="store_true", help="Use LoRA adapter for inference (requires base model + LoRA path)")

    # Save config
    p.add_argument("--out_csv", type=str, default="submission.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    skip_inference = bool(args.skip_inference)
    
    # Validate required arguments based on mode
    if args.mode in ["train", "continue_train"]:
        if not args.train_csv:
            raise ValueError(f"--train_csv is required for mode '{args.mode}'")
        if not args.test_csv and not skip_inference:
            raise ValueError(f"--test_csv is required for mode '{args.mode}' when inference is not skipped")
    elif args.mode in ["inference", "inference_random", "inference_1_temp"]:
        if not args.test_csv:
            raise ValueError(f"--test_csv is required for mode '{args.mode}'")
    
    # Validate LoRA arguments
    if args.use_lora:
        if not args.inference_model_path:
            raise ValueError("--inference_model_path is required when using --use_lora")
        if not args.model_name:
            raise ValueError("--model_name (base model) is required when using --use_lora")
        print(f"==> LoRA mode enabled: base_model={args.model_name}, lora_path={args.inference_model_path}")
    
    if args.mode == "train":
        print("==> Mode: Fresh Training")
        train(args)
        if not skip_inference:
            inference_vllm(args)
    elif args.mode == "continue_train":
        print("==> Mode: Continue Training")
        if os.path.exists(args.out_dir):
            backup = f"{args.out_dir}_backup_{int(time.time())}"
            shutil.copytree(args.out_dir, backup)
            print(f"Backup created: {backup}")
        train(args)
        if not skip_inference:
            inference_vllm(args)
    elif args.mode == "inference":
        print("==> Mode: Inference Only")
        inference_vllm(args)
    elif args.mode == "inference_random":
        print("==> Mode: Inference with Random Temperatures")
        temps = [i * 0.1 for i in range(0, 11)]  # [0.0, 0.1, ..., 1.0]
        for temp in temps:
            csv_name = f"preds_temp_{temp:.1f}.csv"
            print(f"==> Inference with temperature={temp:.1f}")
            inference_vllm(args, temp=temp, force_out_csv=csv_name)
    elif args.mode == "inference_1_temp":
        print("==> Mode: Inference Multiple Runs (temp=0.7)")
        temp = 0.7
        for i in range(10):
            csv_name = f"preds_temp_{temp}_{i + 1}.csv"
            print(f"==> Inference run {i + 1}/10 with temperature={temp}")
            inference_vllm(args, temp=temp, force_out_csv=csv_name)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
