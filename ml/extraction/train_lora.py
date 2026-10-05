import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
import os
try:
    from .dataset import load_and_prepare_medical_dataset
except ImportError:
    try:
        from dataset import load_and_prepare_medical_dataset
    except ImportError:
        from ml.extraction.dataset import load_and_prepare_medical_dataset

def prepare_lora_model(base_model_name: str = "t5-small"):
    """
    Prepares the base T5 model with LoRA adapters for fine-tuning.
    """
    print(f"Loading base model {base_model_name} for training...")
    tokenizer = T5Tokenizer.from_pretrained(base_model_name)
    model = T5ForConditionalGeneration.from_pretrained(base_model_name)
    
    # 1. Setup LoRA Config
    # We target the 'q' and 'v' attention matrices, which is standard practice for Transformers
    lora_config = LoraConfig(
        r=8,                     # Rank of the update matrices (smaller = faster/less memory)
        lora_alpha=32,           # Scaling factor
        target_modules=["q", "v"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # T5 is a Sequence-to-Sequence model
    )
    
    # 2. Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    # This will show you how many parameters you are actually training!
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_t5_lora():
    output_dir = "./lora-medical-t5"
    
    # 1. Prepare Model & Tokenizer
    model, tokenizer = prepare_lora_model()
    
    # 2. Load & Prepare Dataset
    dataset = load_and_prepare_medical_dataset(tokenizer)
    
    # Split into train/eval (80/20 split)
    # Note: Our proxy dataset is small, so we just use the whole thing for this demo
    train_dataset = dataset
    eval_dataset = dataset # In a real scenario, use dataset.train_test_split()
    
    # 3. Setup Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-3,             # LoRA generally uses a higher learning rate
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        load_best_model_at_end=True,
    )
    
    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # 5. Start Training!
    print("Starting LoRA fine-tuning...")
    trainer.train()
    
    # 6. Save the trained LoRA weights
    print(f"Saving trained model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    print("--- Person B: T5 LoRA Training Pipeline ---")
    train_t5_lora()
