import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
    question = """A 52-year-old, nonsmoking man complains of pain and paresthesias in his right hand, paicularly at night. Examination reveals a diminished radial pulse when he abducts his arm when his head is turned to either side. A bruit is audible over the upper right anterior chest. His neurologic examination is unremarkable. You suspect the patient has 
    A. Pancoast tumor 
    B. A cervical rib 
    C. Cervical disc disease 
    D. Subclan steal syndrome
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