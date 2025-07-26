import os
import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Create directories if they don't exist
os.makedirs(os.path.join("..", "data", "raw"), exist_ok=True)

# Disease progression rules (with probabilities)
disease_progression = {
    "hypertension": [("stroke", 0.3), ("heart_disease", 0.4)],
    "diabetes": [("kidney_disease", 0.2), ("neuropathy", 0.3)],
    "flu": [("pneumonia", 0.5)],
}

# Lab test ranges (simulated)
lab_ranges = {
    "glucose": (70, 140),    # mg/dL
    "creatinine": (0.6, 1.2), # mg/dL
    "wbc": (4.5, 11.0),       # 10^3/μL
}

# Medications and their effects
medications = {
    "metformin": {"glucose": -20, "side_effects": ["nausea"]},
    "aspirin": {"wbc": 0, "side_effects": []},
    "insulin": {"glucose": -40, "side_effects": ["hypoglycemia"]},
}

def generate_lab_results(diagnosis):
    labs = {}
    if diagnosis == "diabetes":
        labs["glucose"] = random.randint(140, 300)  # High glucose
    elif diagnosis == "kidney_disease":
        labs["creatinine"] = round(random.uniform(1.5, 5.0), 1)
    else:
        for test, (low, high) in lab_ranges.items():
            labs[test] = round(random.uniform(low, high), 1)
    return labs

def generate_patient(patient_id, num_visits=5):
    records = []
    current_diagnosis = None
    age = random.randint(30, 80)
    current_meds = []
    
    for visit in range(num_visits):
        timestamp = datetime.now() - timedelta(days=random.randint(0, 365*2))
        
        if not current_diagnosis or random.random() < 0.3:
            current_diagnosis = random.choice(list(disease_progression.keys()))
        
        # Generate symptoms (some medication-induced)
        symptoms = random.sample(
            ["fever", "cough", "fatigue", "headache", "chest_pain"],
            k=random.randint(1, 3))
        
        # Add medication side effects
        for med in current_meds:
            symptoms.extend(medications[med]["side_effects"])
        
        # Generate lab results
        labs = generate_lab_results(current_diagnosis)
        
        # Select treatment (with probability)
        if current_diagnosis in ["diabetes", "hypertension"]:
            treatment = random.choice(list(medications.keys()))
            current_meds.append(treatment)
        else:
            treatment = "aspirin"  # Default
        
        records.append({
            "patient_id": patient_id,
            "age": age,
            "timestamp": timestamp.strftime("%Y-%m-%d"),
            "symptoms": symptoms,
            "diagnosis": current_diagnosis,
            "treatment": treatment,
            **labs  # Unpack lab results
        })
        
        # Simulate disease progression
        if current_diagnosis in disease_progression:
            for (next_diag, prob) in disease_progression[current_diagnosis]:
                if random.random() < prob:
                    current_diagnosis = next_diag
                    break
    
    return pd.DataFrame(records)

# Generate dataset - THIS COMES AFTER ALL FUNCTION DEFINITIONS
all_patients = pd.concat([generate_patient(f"P{i:03d}") for i in range(100)])
all_patients.to_csv(os.path.join("..", "data", "raw", "synthetic_ehr.csv"), index=False)
print("Data generated successfully at ../data/raw/synthetic_ehr.csv")