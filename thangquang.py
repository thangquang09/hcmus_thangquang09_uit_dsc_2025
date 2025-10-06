import unsloth
import argparse
import os
import shutil
import time
import json

from vllm import LLM, SamplingParams
import pandas as pd
import torch
from huggingface_hub import create_repo, upload_folder
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from tqdm import tqdm
import random
from collections import defaultdict


from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling

from huggingface_hub import login

# Login will be handled by args.hf_token later



class BoolMaskCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if "attention_mask" in batch:
            mask = batch["attention_mask"]
            if mask.dtype != torch.bool:
                batch["attention_mask"] = mask.bool()
        return batch



# ==========================
# Utils
# ==========================

fewshot_json_path = "data/data_fewshot_final.json"
with open(fewshot_json_path, "r") as f:
    fewshot_data = json.load(f)

LABELS = ["no", "intrinsic", "extrinsic"]

def sample_fewshots(fewshot_data, k=5, seed=None):
    """
    Lấy k ví dụ few-shot với điều kiện:
      - có đủ cả 3 nhãn trong LABELS (mỗi nhãn >= 1)
      - phần còn lại chọn ngẫu nhiên từ toàn bộ pool (trừ các mẫu đã lấy)
      - trộn ngẫu nhiên thứ tự kết quả
    """
    if k < len(LABELS):
        raise ValueError(f"k={k} phải >= số nhãn {len(LABELS)}")

    # Gom theo nhãn
    by_label = defaultdict(list)
    for ex in fewshot_data:
        lab = ex.get("label")
        if lab in LABELS:
            by_label[lab].append(ex)

    # Kiểm tra đủ nguồn mỗi nhãn
    missing = [lab for lab in LABELS if len(by_label[lab]) == 0]
    if missing:
        raise ValueError(f"fewshot_data thiếu ví dụ cho các nhãn: {missing}")

    rng = random.Random(seed)

    # Bước 1: đảm bảo phủ đủ 3 nhãn (mỗi nhãn chọn 1)
    selected = [rng.choice(by_label[lab]) for lab in LABELS]

    # Bước 2: chọn thêm (k-3) mẫu ngẫu nhiên từ phần còn lại
    used_ids = set(map(id, selected))
    remaining_pool = [ex for ex in fewshot_data if id(ex) not in used_ids]

    if len(remaining_pool) < (k - len(LABELS)):
        raise ValueError(f"Không đủ few-shots để lấy {k} mẫu (pool còn {len(remaining_pool)})")

    rng.shuffle(remaining_pool)
    selected.extend(remaining_pool[:(k - len(LABELS))])

    # Bước 3: trộn thứ tự kết quả để tránh bias vị trí
    rng.shuffle(selected)
    return selected


def free_gpu():
    """Enhanced GPU memory cleanup"""
    import gc
    import torch

    # Force garbage collection multiple times
    for _ in range(5):  # Tăng số lần GC
        gc.collect()

    if torch.cuda.is_available():
        # Clear all cached memory
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Force clear all devices
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Memory info
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(
            f"[INFO] GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
        )

    print("[INFO] GPU memory cleaned!")



def norm(x):
    return "" if x is None or str(x).lower() == "nan" else str(x)


def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    for lab in ["extrinsic", "intrinsic", "no"]:
        if s.startswith(lab) or lab in s:
            return lab
    return "no"


def build_few_shots(fewshot_data):
    FEWSHOT = "Context: {context}\n\nPrompt: {prompt}\n\nResponse: {response}\n\nLabel: {label}\n\nExplanation: {explanation}\n\n"
    result = ""
    for data in fewshot_data:
        result += FEWSHOT.format(
            context=data["context"],
            prompt=data["prompt"],
            response=data["generated_response"],
            label=data["label"],
            explanation=data["explanation"],
        )
    return result


def build_user_msg_train(context, prompt, response, fewshot_data):
    INSTRUCTION = """You are a hallucination detection classifier for Vietnamese language models. 
Your task is to classify the RESPONSE into exactly ONE label from {no, intrinsic, extrinsic}, 
based ONLY on the given CONTEXT and PROMPT. 
You must NEVER use knowledge outside the provided CONTEXT.

Label Definitions:
- no: RESPONSE is fully supported by CONTEXT, with no added or fabricated content. 
       Allowed to reject false assumptions in PROMPT if CONTEXT shows they are wrong.
- intrinsic: RESPONSE contradicts, reverses, or distorts facts from CONTEXT. 
             This includes repeating false assumptions from PROMPT that conflict with CONTEXT.
- extrinsic: RESPONSE adds new information not grounded in CONTEXT and not directly verifiable from it, 
             without explicit contradiction.

Classification Rules:
1) If RESPONSE both contradicts CONTEXT AND adds unsupported info → intrinsic (contradiction takes priority).
2) Match at semantic level; ignore minor spelling or grammatical errors.
3) If PROMPT contains false assumptions and RESPONSE accepts/repeats them against CONTEXT → intrinsic.
4) If RESPONSE only says “insufficient / not enough information” (without fabricating) → no.
5) Output must be EXACTLY one word: no | intrinsic | extrinsic

Evaluation Order:
1. First check for contradictions with CONTEXT → intrinsic
2. If no contradiction, check for unsupported additions → extrinsic
3. If fully supported with no addition → no
"""

    FEWSHOT = """EXAMPLE CLASSIFICATION:\n\n\n""" + build_few_shots(fewshot_data)

    return (
        INSTRUCTION
        + "\n\n"
        + FEWSHOT
        + "\n\n"
        + f"Please classify the following samples:\n\nContext: {context}\n\n"
        f"Prompt: {prompt}\n\n"
        f"Response: {response}\n"
        f"Label:"
    )


def build_assistant_msg_train(label):
    return f"{normalize_label(label)}"


# ==========================
# Dataset
# ==========================
def build_train_dataset(train_csv, tokenizer, mock=False):
    from datasets import Dataset

    df = pd.read_csv(train_csv)

    if mock:
        df = df.head(5)

    required = ["id", "context", "prompt", "response", "label"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Thiếu cột {col}")

    def row_to_conv(row):
        fewshot_subset = sample_fewshots(fewshot_data, k=5, seed=str(row["id"]))
        user_msg = build_user_msg_train(
            row["context"], row["prompt"], row["response"], fewshot_subset
        )  # ✅ Thêm fewshot_data parameter
        assistant_msg = build_assistant_msg_train(row["label"])
        return {
            "conversations": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }

    ds = Dataset.from_pandas(df).map(row_to_conv)

    def to_text(examples):
        texts = [
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            for conv in examples["conversations"]
        ]
        # Debug: in ra sample đầu tiên
        # if len(texts) > 0:
        #     print(f"==> Sample formatted text:\n{texts[0][:200]}...")
        return {"text": texts}

    final_ds = ds.map(to_text, batched=True)
    print(f"==> Dataset processed: {len(final_ds)} samples")
    return final_ds


def train(args):
    # Setting
    free_gpu()
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        login(token=args.hf_token)
        print("==> Logged in to HuggingFace")

    if args.load_in_8bit:
        print("==> Using 8-bit quantization")
    else:
        print("==> Using 4-bit quantization")

    # Continue training với Unsloth
    if args.mode == "continue_train":
        # Xác định checkpoint path để load
        if args.train_from:
            checkpoint_path = args.train_from
            print(f"==> Continue training from specified checkpoint: {checkpoint_path}")
        else:
            checkpoint_path = args.out_dir
            print(f"==> Continue training from default checkpoint: {checkpoint_path}")

        # Kiểm tra xem checkpoint có phải là HuggingFace repo hay local path
        is_hf_repo = not os.path.exists(checkpoint_path) and "/" in checkpoint_path
        
        if os.path.exists(checkpoint_path) or is_hf_repo:
            try:
                # Kiểm tra xem model có phải là LoRA model không
                is_lora_model = False
                if os.path.exists(checkpoint_path):
                    # Check local path
                    is_lora_model = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
                else:
                    # Check HF repo
                    try:
                        from huggingface_hub import hf_hub_download
                        hf_hub_download(repo_id=checkpoint_path, filename="adapter_config.json", token=args.hf_token)
                        is_lora_model = True
                    except Exception:
                        is_lora_model = False
                
                if is_lora_model:
                    # Load LoRA checkpoint
                    print("==> Detected LoRA model, loading with LoRA adapters...")
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=checkpoint_path,
                        max_seq_length=args.max_seq_len,
                        load_in_4bit=not args.load_in_8bit,
                        load_in_8bit=args.load_in_8bit,
                        token=args.hf_token,
                    )
                    print(f"==> Successfully loaded LoRA checkpoint from {checkpoint_path}")
                else:
                    # Load full merged model và apply LoRA
                    print("==> Detected full model, loading and applying new LoRA adapters...")
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=checkpoint_path,
                        max_seq_length=args.max_seq_len,
                        load_in_4bit=not args.load_in_8bit,
                        load_in_8bit=args.load_in_8bit,
                        token=args.hf_token,
                    )
                    
                    # Apply LoRA adapters to the full model
                    model = FastLanguageModel.get_peft_model(
                        model,
                        r=args.lora_r,
                        target_modules=[
                            "q_proj",
                            "k_proj",
                            "v_proj",
                            "o_proj",
                            "gate_proj",
                            "up_proj",
                            "down_proj",
                        ],
                        lora_alpha=args.lora_alpha,
                        lora_dropout=args.lora_dropout,
                        use_gradient_checkpointing="unsloth",
                        random_state=args.seed,
                    )
                    print(f"==> Successfully loaded full model from {checkpoint_path} and applied LoRA")
                    
            except Exception as e:
                print(f"ERROR: Failed to load checkpoint {checkpoint_path}: {str(e)}")
                print(f"==> Starting fresh training with base model: {args.model_name}")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=args.model_name,
                    max_seq_length=args.max_seq_len,
                    load_in_4bit=not args.load_in_8bit,
                    load_in_8bit=args.load_in_8bit,
                    token=args.hf_token,
                )

                model = FastLanguageModel.get_peft_model(
                    model,
                    r=args.lora_r,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    lora_alpha=args.lora_alpha,
                    lora_dropout=args.lora_dropout,
                    use_gradient_checkpointing="unsloth",
                    random_state=args.seed,
                )
        else:
            print(
                f"WARNING: Checkpoint {checkpoint_path} not found, starting fresh training"
            )
            print(f"==> Loading base model: {args.model_name}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model_name,
                max_seq_length=args.max_seq_len,
                load_in_4bit=not args.load_in_8bit,
                load_in_8bit=args.load_in_8bit,
                token=args.hf_token,
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_r,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )

    else:
        print("==> Fresh training mode")
        print(f"==> Loading base model: {args.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_len,
            load_in_4bit=not args.load_in_8bit,
            load_in_8bit=args.load_in_8bit,
            token=args.hf_token,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    # Apply chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-instruct",
    )

    # Load train dataset
    mock = bool(args.mock)  # Convert to boolean
    train_ds = build_train_dataset(args.train_csv, tokenizer, mock)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=BoolMaskCollator(tokenizer),
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            # warmup_steps=args.warmup_steps,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr * 0.5 if args.mode == "continue_train" else args.lr,
            logging_steps=args.logging_steps,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=args.save_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            output_dir=args.out_dir,
        ),
    )

    # Khởi tạo trainer với chat template
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    # Bắt đầu train
    stats = trainer.train()
    print("==> Train done:", stats.metrics.get("train_runtime"))

    # Save model (phần này chỉ save Lora Config)
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # Save for VLLM: Merge 16bit
    model.save_pretrained_merged(
        f"{args.out_dir}_vllm", tokenizer, save_method="merged_16bit"
    )


# ==========================
# Inference
# ==========================
def inference_vllm(args, temp=None, force_out_csv=None):
    """Inference với model đã train hoặc base model"""
    print("==> Inference...")
    free_gpu()
    
    # Login to HuggingFace if token provided
    if args.hf_token:
        login(token=args.hf_token)
        print("==> Logged in to HuggingFace for inference")
    
    sampling_params = SamplingParams(
        temperature=0.7 if temp is None else temp,
        top_p=0.8,
        top_k=20,
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
        gpu_memory_utilization=0.8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=args.hf_token)
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
    df = df[["id", "predict_label", "raw_output"]]
    
    out_csv = force_out_csv if force_out_csv else args.out_csv
    
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"==> Inference done with vLLM! Saved to {out_csv}")


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
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--train_from", type=str, help="Path or HuggingFace repo to continue training from")
    p.add_argument("--out_dir", type=str, default="lora_model")
    p.add_argument("--repo_id", type=str, default="thang09/uit_qwen3_thinking")
    p.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Instruct-2507")
    p.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for accessing private models")
    p.add_argument(
        "--max_seq_len", type=int, default=8096
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

    # Save config
    p.add_argument("--out_csv", type=str, required=True, default="submisison.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    skip_inference = bool(args.skip_inference)

    # Validation: inference mode requires inference_model_path
    if args.mode == "inference" and not args.inference_model_path:
        raise ValueError("--inference_model_path is required when mode is 'inference'")

    if args.mode == "train":
        print("==> Mode: Fresh Training")
        train(args)
        print("==> Training Completed")
        if not skip_inference:
            inference_vllm(args)
    elif args.mode == "continue_train":
        print("==> Mode: Continue Training")

        # Backup model ở out_dir trước khi continue (chỉ khi model tồn tại)
        # Note: Chỉ backup out_dir, không backup train_from
        if os.path.exists(args.out_dir):
            backup_dir = f"{args.out_dir}_backup_{int(time.time())}"
            shutil.copytree(args.out_dir, backup_dir)
            print(f"==> Backup created at: {backup_dir}")

        # Hiển thị thông tin về checkpoint source
        if args.train_from:
            print(f"==> Will load from: {args.train_from}")
            print(f"==> Will save to: {args.out_dir}")
        else:
            print(f"==> Will load and save from/to: {args.out_dir}")

        train(args)
        print("==> Continue training completed")
        if not skip_inference:
            inference_vllm(args)
    elif args.mode == "inference":
        print("==> Mode: Inference Only")
        inference_vllm(args)
    elif args.mode == "inference_random":
        temps = [i * 0.1 for i in range(0, 11)]  # [0.0, 0.1, ..., 1.0]
        for temp in temps:
            csv_name = f"preds_qwen4b_temp_{temp:.1f}.csv"
            print(f"==> Inference with temperature={temp:.1f}")
            inference_vllm(args, temp, csv_name)
    elif args.mode == "inference_1_temp":
        temp = 0.7
        for i in range(10):
            csv_name = f"preds_qwen4b_temp_{temp}_{i + 1}.csv"
            print(f"==> Inference with temperature={temp:.1f}")
            inference_vllm(args, temp, csv_name)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
