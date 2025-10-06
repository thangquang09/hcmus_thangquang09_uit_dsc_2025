import unsloth
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer
from src.data_utils.load_data import TrainDatasetBuilder
from src.utils.free_gpu import free_gpu
from src.utils.boolmaskcollator import BoolMaskCollator
from src.model.qwen3_instruct import load_model
from src.utils.constants import FEWSHOT_PATH
import json


def train(args):
    free_gpu()
    with open(FEWSHOT_PATH, "r") as f:
        fewshot_data = json.load(f)

    # Sử dụng train_from nếu có, nếu không thì None (load base model)
    continue_path = getattr(args, 'train_from', None) if hasattr(args, 'train_from') else None
    model, tokenizer = load_model(args, continue_path=continue_path)
    builder = TrainDatasetBuilder(tokenizer, fewshot_data, mock=bool(args.mock), fewshot_k=5)
    train_ds = builder.build(args.train_csv)
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        data_collator=BoolMaskCollator(tokenizer),
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            output_dir=args.out_dir,
            logging_steps=args.logging_steps,
            weight_decay=args.weight_decay,
            seed=args.seed,
            save_strategy="steps",
            save_steps=args.save_steps,
        ),
    )

    trainer = train_on_responses_only(trainer, "<|im_start|>user\n", "<|im_start|>assistant\n")
    stats = trainer.train()
    print("==> Train done:", stats.metrics.get("train_runtime"))

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    model.save_pretrained_merged(f"{args.out_dir}_vllm", tokenizer, save_method="merged_16bit")
