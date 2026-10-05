import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import PeftModel
from typing import Dict, Any

class MedicalExtractor:
    def __init__(self, base_model_name: str = "t5-small", lora_weights_path: str = None):
        """
        Initializes the T5 model for medical extraction.
        If lora_weights_path is provided, it loads the fine-tuned LoRA weights.
        """
        print(f"Loading base model: {base_model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(base_model_name)
        
        if lora_weights_path:
            print(f"Loading LoRA weights from {lora_weights_path}")
            # Load the LoRA adapter onto the base model
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate_prompt(self, transcript: str) -> str:
        """
        Creates the prompt to feed into the T5 model.
        T5 is an encoder-decoder model, so we give it a clear task prefix.
        """
        prompt = (
            "extract medical information: "
            f"Transcript: {transcript} "
            "Output format: Symptoms: | Diagnosis: | Prescriptions: | Follow-up: "
        )
        return prompt

    def extract(self, transcript: str) -> Dict[str, str]:
        """
        Runs the extraction on the provided transcript.
        """
        prompt = self.generate_prompt(transcript)
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=4,
                early_stopping=True
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_output(generated_text)
        
    def _parse_output(self, generated_text: str) -> Dict[str, str]:
        """
        A simple parser to convert the text output into a structured dictionary.
        This expects the model to output something like:
        "Symptoms: Fever | Diagnosis: Viral Infection | Prescriptions: Paracetamol | Follow-up: 3 days"
        """
        result = {
            "symptoms": "",
            "diagnosis": "",
            "prescriptions": "",
            "follow_up": "",
            "raw_output": generated_text
        }
        
        # Very basic parsing based on the expected prompt format
        parts = generated_text.split('|')
        for part in parts:
            part = part.strip()
            if part.startswith("Symptoms:"):
                result["symptoms"] = part.replace("Symptoms:", "").strip()
            elif part.startswith("Diagnosis:"):
                result["diagnosis"] = part.replace("Diagnosis:", "").strip()
            elif part.startswith("Prescriptions:"):
                result["prescriptions"] = part.replace("Prescriptions:", "").strip()
            elif part.startswith("Follow-up:"):
                result["follow_up"] = part.replace("Follow-up:", "").strip()
                
        return result
