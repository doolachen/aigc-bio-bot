import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from trl import SFTTrainer
from unsloth import FastLanguageModel,is_bf16_supported
from transformers import TrainingArguments
from dataset.gen_dataset import gen_dataset

MAX_SEQ_LENTH = 4096

if __name__ == '__main__':
    path = "models/DeepSeek-R1-Distill-Qwen-32B"
    print('Load model...')
    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=path,
            max_seq_length=MAX_SEQ_LENTH,
            dtype=None,
            load_in_4bit=True, # 4-bit quant
            local_files_only=True
    )
    print('Load dataset...')
    dataset_name = "FreedomIntelligence/medical-o1-reasoning-SFT"
    train_dataset = gen_dataset(dataset_name)
    print(train_dataset)
    
    lora_model = FastLanguageModel.get_peft_model(
        model,
        r=32,   # rank = 32 :high Adaptability
        lora_alpha=64, # r x 2
        lora_dropout=0,  
        use_gradient_checkpointing=True,
        random_state=1086,
        loftq_config=None,
    )
    
    trainer = SFTTrainer(
        model=lora_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENTH,
        dataset_num_proc=1,
        args=TrainingArguments(
            save_steps=100,
            save_total_limit=3,
            per_device_train_batch_size=12,  
            num_train_epochs=3,
            warmup_steps=5,
            max_steps=-1,
            learning_rate=2e-4,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=1086,
            output_dir="outputs",
            report_to="tensorboard"
        ),
    )
    trainer_stats = trainer.train(resume_from_checkpoint = False)
    # Save Lora config
    lora_model.save_pretrained_merged("models/DeepSeek-R1-Distill-Qwen-32B-Lora", tokenizer, save_method = "lora",)
    
    # Save Full model
    lora_model.save_pretrained("models/DeepSeek-R1-Distill-Qwen-32B-Lora-Full") 
    tokenizer.save_pretrained("models/DeepSeek-R1-Distill-Qwen-32B-Lora-Full")
    lora_model.save_pretrained_merged("models/DeepSeek-R1-Distill-Qwen-32B-Lora-Full", tokenizer, save_method = "merged_16bit",)