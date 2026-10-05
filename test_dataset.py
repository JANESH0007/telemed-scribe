from datasets import load_dataset
try:
    print("Trying to load FreedomIntelligence/MTS-Dialog...")
    ds = load_dataset("FreedomIntelligence/MTS-Dialog", split="train[:5]")
    print(ds.column_names)
    print(ds[0])
except Exception as e:
    print("Failed MTS-Dialog:", e)

try:
    print("\nTrying to load lavita/ChatDoctor...")
    ds2 = load_dataset("lavita/ChatDoctor", split="train[:5]")
    print(ds2.column_names)
    print(ds2[0])
except Exception as e:
    print("Failed ChatDoctor:", e)
