from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

def load_and_prepare_medical_dataset(tokenizer: PreTrainedTokenizer, max_source_length=512, max_target_length=256) -> Dataset:
    """
    Loads a medical dialogue dataset and prepares it for T5 fine-tuning.
    This module simulates loading the MTS-Dialog benchmark and formats it 
    into T5 instruction-target pairs.
    """
    print("Loading medical dialogue dataset...")
    
    # =====================================================================
    # NOTE: Most high-quality medical datasets on Hugging Face (like MTS-Dialog)
    # are now 'Gated' due to medical privacy rules (HIPAA). 
    # For a student project, the standard industry practice is to curate your 
    # own custom JSON dataset containing doctor-patient scenarios!
    #
    # We will load from our custom 'medical_dataset.json' file:
    # =====================================================================
    
    # Load our local JSON dataset!
    import os
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "medical_dataset.json")
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def preprocess_function(examples):
        inputs = []
        targets = []
        
        # We loop through the batch and format our Prompts & Targets
        for i in range(len(examples["dialogue"])):
            # 1. Format the Input Prompt (Must match the format in extractor.py!)
            dialogue = examples["dialogue"][i]
            input_prompt = f"extract medical information: Transcript: {dialogue} Output format: Symptoms: | Diagnosis: | Prescriptions: | Follow-up: "
            inputs.append(input_prompt)
            
            # 2. Format the Target Output
            sym = examples["symptoms"][i]
            diag = examples["diagnosis"][i]
            rx = examples["prescriptions"][i]
            fup = examples["follow_up"][i]
            target_text = f"Symptoms: {sym} | Diagnosis: {diag} | Prescriptions: {rx} | Follow-up: {fup}"
            targets.append(target_text)
            
        # Tokenize the inputs and targets
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding="max_length", truncation=True)
        labels = tokenizer(targets, max_length=max_target_length, padding="max_length", truncation=True)
        
        # Crucial Step for T5 Training: 
        # Replace the padding token id with -100 so it is ignored in the Loss computation
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] 
            for label in labels["input_ids"]
        ]
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing and mapping dataset to T5 format...")
    # Apply the preprocessing to the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
    
    return tokenized_dataset
