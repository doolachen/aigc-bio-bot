from transformers import AutoTokenizer
from datasets import load_dataset,load_from_disk
import os 
# template
DATASET_TEMPLATE = """
    Below is a task description paired with additional context. Your assignment is to craft a response that fully addresses the request. 
    Prior to providing your final answer, carefully analyze the question and outline a step-by-step reasoning process (chain of thought) to ensure that your final response is both logical and precise.
    ### Instruction:
    You are a distinguished medical professional with extensive expertise in clinical diagnostics, patient assessment, and treatment strategy formulation. Your task is to answer the following medical inquiry.

    ### Question:
    {}
    ### Response:
    <think>
    {}
    </think>
    {}
"""

model_path = "models/DeepSeek-R1-Distill-Qwen-32B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
END_TOKEN = tokenizer.eos_token
    

def process_data(records):
    queries = records["Question"]
    reasoning_steps = records["Complex_CoT"]
    replies = records["Response"]
    formatted_texts = []
    
    for query, reasoning, reply in zip(queries, reasoning_steps, replies):
        formatted_entry = DATASET_TEMPLATE.format(query, reasoning, reply) + END_TOKEN
        formatted_texts.append(formatted_entry)
    
    return {"text": formatted_texts}


def gen_dataset(dataset_name):
    local_path = "./dataset/medical-o1-reasoning-dataset"
    if os.path.exists(local_path):
        print("Loading dataset from local storage...")
        dataset = load_from_disk(local_path)
    else:
        print("Downloading dataset from Hugging Face...")
        dataset = load_dataset(dataset_name, "en", split="train[0:24000]", trust_remote_code=True)
        dataset = dataset.map(process_data, batched=True)
        os.makedirs(local_path, exist_ok=True)
        dataset.save_to_disk(local_path)
        print(f"Dataset saved to {local_path}.")

    return dataset

if __name__ == '__main__':
    dataset_name = "FreedomIntelligence/medical-o1-reasoning-SFT"
    dataset = gen_dataset(dataset_name)
    print(dataset)
    