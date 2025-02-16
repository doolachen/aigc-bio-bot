from huggingface_hub import snapshot_download

if __name__ == '__main__':
    model_name = "unsloth/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
    save_path = "./models/DeepSeek-R1-Distill-Qwen-32B"
    snapshot_download(repo_id=model_name, local_dir=save_path)
