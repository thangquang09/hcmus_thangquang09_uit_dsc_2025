import os
from huggingface_hub import HfApi, create_repo, upload_folder

def push_to_hf(
    model_dir="lora_model", 
    repo_id="thangquang09/uit_qwen3_instruct_lora64_vllm", 
    commit_msg="Upload fine-tuned model"
):
    # Lấy token từ biến môi trường
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Bạn cần export HF_TOKEN trước khi chạy!")

    # Tạo repo nếu chưa có
    print(f"==> Creating repo {repo_id} (nếu đã có thì bỏ qua)...")
    create_repo(repo_id, token=token, exist_ok=True, private=False)

    # Upload folder (bao gồm cả tokenizer + config)
    print(f"==> Uploading {model_dir} to HF Hub...")
    upload_folder(
        repo_id=repo_id,
        folder_path=model_dir,
        commit_message=commit_msg,
        token=token,
    )
    print("==> Done! Model đã được push lên Hugging Face.")


if __name__ == "__main__":
    push_to_hf(
        model_dir="uit_dsc_lora_model_final_vllm",   # đổi sang folder bạn muốn push
        repo_id="thangquang09/uit_dsc_lora_model_final_vllm",  # repo id của bạn
        commit_msg="Upload model"
    )