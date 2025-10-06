import unsloth

from transformers import DataCollatorForLanguageModeling
import torch    
class BoolMaskCollator(DataCollatorForLanguageModeling):
    """Collator để fix lỗi attention_mask dtype"""
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        if "attention_mask" in batch:
            mask = batch["attention_mask"]
            if mask.dtype != torch.bool:
                batch["attention_mask"] = mask.bool()
        return batch