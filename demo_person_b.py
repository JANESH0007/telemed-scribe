from ml.extraction import MedicalExtractor

def main():
    print("--- Person B Demo: T5 Medical Extraction ---")
    
    # 1. Initialize your model (Now with the trained LoRA weights!)
    extractor = MedicalExtractor(
        base_model_name="t5-small", 
        lora_weights_path="./lora-medical-t5"
    )
    
    # 2. Simulate Person A's output (The clean English transcript)
    sample_transcript = (
        "Patient complains of a severe headache and fever for the last 3 days. "
        "Diagnosed with a suspected viral infection. "
        "Prescribed Paracetamol 500mg to be taken twice a day after meals. "
        "Advised to drink plenty of fluids and rest. "
        "Follow up in 5 days if symptoms do not improve."
    )
    
    print("\n[Input from Person A]")
    print(f"Transcript: {sample_transcript}")
    
    print("\n[Running Extractor...]")
    # 3. Extract the details!
    results = extractor.extract(sample_transcript)
    
    print("\n[Extracted Details]")
    for key, value in results.items():
        if key != "raw_output":
            print(f"- {key.capitalize()}: {value}")
            
    print("\n[Raw T5 Output]")
    print(results["raw_output"])
    
    print("\nNote: Since this is using 'vanilla' t5-small (without your LoRA training yet), ")
    print("the output format might be messy or incorrect. That's why you need to train it!")

if __name__ == "__main__":
    main()
