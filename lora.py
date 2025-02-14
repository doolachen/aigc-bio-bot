import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from unsloth import FastLanguageModel

if __name__ == '__main__':
    path = "./models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-32B/snapshots/3865e12a1eb7cbd641ab3f9dfc28c588c6b0c1e9"
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=4000,
            dtype=None,
            load_in_4bit=True,
            local_files_only=True,
            device_map="auto"
        )
    
    FastLanguageModel.for_inference(model)
    # model.to(torch.device("cuda:3")) 
    print("Model loaded successfully!")
    
    # System prompt
    prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
    Write a response that appropriately completes the request. 
    Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

    ### Instruction:
    You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
    Please answer the following medical question. 

    ### Question:
    {}

    ### Response:
    <think>{}"""
    
    # Test question
    question = """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or 
                sneezing but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings, 
                what would cystometry most likely reveal about her residual volume and detrusor contractions?"""
    
    
    # Format the question using the structured prompt (`prompt_style`) and tokenize it
    inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

    # Generate a response using the model
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )

    response = tokenizer.batch_decode(outputs)
    print(response[0].split("### Response:")[1])  