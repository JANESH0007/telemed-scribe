import json
import random

symptoms_list = [
    ("fever and chills", "viral fever"),
    ("severe headache", "migraine"),
    ("persistent cough", "bronchitis"),
    ("sore throat and difficulty swallowing", "strep throat"),
    ("nausea and vomiting", "gastroenteritis"),
    ("lower back pain", "muscle strain"),
    ("heartburn after meals", "acid reflux"),
    ("dizziness and high BP", "hypertension"),
    ("wheezing and shortness of breath", "asthma exacerbation"),
    ("sneezing and runny nose", "allergic rhinitis"),
    ("itchy red rash", "contact dermatitis"),
    ("sharp pain down the leg", "sciatica"),
    ("constant fatigue", "anemia"),
    ("weight gain and tiredness", "hypothyroidism"),
    ("joint pain in knees", "osteoarthritis")
]

medications = [
    "Paracetamol 500mg", "Ibuprofen 400mg", "Amoxicillin 500mg", 
    "Omeprazole 20mg", "Amlodipine 5mg", "Albuterol inhaler", 
    "Loratadine 10mg", "Hydrocortisone cream 1%", "Naproxen 500mg", 
    "Iron supplements", "Levothyroxine 50mcg", "Diclofenac gel",
    "Azithromycin 250mg", "Cetirizine 10mg"
]

follow_ups = ["5 days", "1 week", "2 weeks", "1 month", "None specified", "If symptoms worsen", "3 days", "10 days"]

def generate_dialogue(symptom, diagnosis, med, follow_up):
    templates = [
        f"Doctor: What brings you in today? Patient: I've been dealing with {symptom} for a few days. Doctor: Based on your exam, it looks like {diagnosis}. I am prescribing {med}. Patient: Do I need to come back? Doctor: Follow up in {follow_up}.",
        f"Patient: I have {symptom}. It's really bothering me. Doctor: Let me take a look. Yes, this is clearly {diagnosis}. Take {med} for relief. Come back in {follow_up}.",
        f"Doctor: How are you feeling? Patient: Not great, I have {symptom}. Doctor: I suspect it's {diagnosis}. I'll write a prescription for {med}. Follow up: {follow_up}.",
        f"Patient: My main issue is {symptom}. Doctor: I understand. That is common with {diagnosis}. Let's start you on {med}. We will review your progress in {follow_up}."
    ]
    return random.choice(templates)

dataset = []

print("Generating 1000 synthetic clinical encounters...")
for _ in range(1000):
    # Pick random attributes
    sym_diag = random.choice(symptoms_list)
    symptom = sym_diag[0]
    diagnosis = sym_diag[1]
    med = random.choice(medications)
    follow_up = random.choice(follow_ups)
    
    # Generate dialogue string
    dialogue = generate_dialogue(symptom, diagnosis, med, follow_up)
    
    # Create structured object
    entry = {
        "dialogue": dialogue,
        "symptoms": symptom,
        "diagnosis": diagnosis,
        "prescriptions": med,
        "follow_up": follow_up
    }
    dataset.append(entry)

# Save to JSON
with open("medical_dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print("Successfully generated medical_dataset.json with 1000 samples!")
