from huggingface_hub import snapshot_download

if __name__ == '__main__':
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    save_path = "./"
    snapshot_download(repo_id=model_name, cache_dir=save_path)
