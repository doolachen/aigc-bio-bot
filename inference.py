import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
from unsloth import FastLanguageModel

if __name__ == '__main__':
    path = "models/DeepSeek-R1-Distill-Qwen-32B"
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=4000,
            dtype=None,
            load_in_4bit=True, # 4-bit quant
            local_files_only=True,
            device_map="auto"
        )
    
    FastLanguageModel.for_inference(model)
    # model.to(torch.device("cuda:3")) 
    print("Model loaded successfully!")
    
    # System prompt
    prompt_style = """Below is a task description paired with additional context. Your assignment is to craft a response that fully addresses the request. 
    Prior to providing your final answer, carefully analyze the question and outline a step-by-step reasoning process (chain of thought) to ensure that your final response is both logical and precise.

    ### Instruction:
    You are a distinguished medical professional with extensive expertise in clinical diagnostics, patient assessment, and treatment strategy formulation. Your task is to answer the following medical inquiry.
    
    ### Question:
    {}

    ### Response:
    <think>{}"""
    
    # Test question
    question = """A 45-year-old man with a long history of chronic alcohol use presents to the emergency department with severe epigastric pain radiating to his back, accompanied by nausea and vomiting. On physical examination, he appears distressed, and his abdomen is tender with guarding. Laboratory results reveal significantly elevated serum amylase and lipase levels.
    Based on these findings, what is the most likely diagnosis, what potential complications should be anticipated, and what would be the recommended management strategy?
    """
    
    
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