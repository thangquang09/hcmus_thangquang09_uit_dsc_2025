import unsloth

import pandas as pd
from datasets import Dataset
from src.prompt.prompt_utils import (
    build_user_msg_train,
    build_assistant_msg_train,
    sample_fewshots,
)
from src.utils.constants import LABELS

class TrainDatasetBuilder:
    """
    Dùng để load, map và build dataset từ CSV sang định dạng training cho SFTTrainer.
    """

    def __init__(self, tokenizer, fewshot_data, mock=False, fewshot_k=5):
        """
        Args:
            tokenizer: tokenizer dùng để apply chat template.
            fewshot_data: danh sách các ví dụ fewshot (list[dict])
            mock: nếu True -> chỉ lấy vài sample đầu (debug)
            fewshot_k: số lượng fewshots mỗi sample (default 5)
        """
        self.tokenizer = tokenizer
        self.fewshot_data = fewshot_data
        self.mock = mock
        self.fewshot_k = fewshot_k

    def _validate_columns(self, df):
        required = ["id", "context", "prompt", "response", "label"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Thiếu cột bắt buộc: {col}")

    def _row_to_conversation(self, row):
        """Tạo đối tượng conversation cho 1 sample"""
        fewshot_subset = sample_fewshots(
            self.fewshot_data, k=self.fewshot_k, seed=str(row["id"])
        )
        user_msg = build_user_msg_train(
            row["context"], row["prompt"], row["response"], fewshot_subset
        )
        assistant_msg = build_assistant_msg_train(row["label"])
        return {
            "conversations": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }

    def _convert_to_text(self, examples):
        """Áp dụng chat template để biến conversation → text"""
        texts = [
            self.tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=False
            )
            for conv in examples["conversations"]
        ]
        return {"text": texts}

    def build(self, csv_path):
        """
        Xây dựng dataset hoàn chỉnh cho training.
        Returns:
            datasets.Dataset
        """
        df = pd.read_csv(csv_path)
        if self.mock:
            df = df.head(5)

        self._validate_columns(df)

        ds = Dataset.from_pandas(df).map(self._row_to_conversation)
        final_ds = ds.map(self._convert_to_text, batched=True)

        print(f"[INFO] ✅ Dataset processed: {len(final_ds)} samples")
        return final_ds
