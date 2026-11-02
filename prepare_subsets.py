import os
from datasets import load_dataset
from collections import Counter

# 1. Configuration
# This is the directory train.py will look inside
BASE_OUTPUT_PATH = "./balanced_training_subsets"  
DATASET_NAME = "QIAIUNCC/OCT-summary-Dataset"
LOCAL_DATASET = "/data"

# Exact sizes train.py looks for by default (capped at ~27k for public data)
TRAINING_SIZES = [100, 248, 500, 748, 1000, 2000, 5000, 10000, 15000, 20000, 26000]

# 2. Load the full public dataset
print("Loading full dataset from Hugging Face...")
full_dataset = load_dataset(DATASET_NAME, split="train")
print(f"Total available samples: {len(full_dataset)}")

full_dataset = full_dataset.class_encode_column('label')

# 3. Create base directory
os.makedirs(BASE_OUTPUT_PATH, exist_ok=True)

# 4. Generate stratified subsets
for size in TRAINING_SIZES:
    # Cap the size at the total available dataset length to prevent errors
    actual_size = min(size, len(full_dataset))
    subset_dir = os.path.join(BASE_OUTPUT_PATH, str(actual_size))
    
    if os.path.exists(subset_dir):
        print(f"Subset {actual_size} already exists at {subset_dir}. Skipping.")
        continue
        
    print(f"\nGenerating stratified subset of size {actual_size}...")
    
    # Stratified split: we split off the amount we DON'T want, keeping the remainder.
    # This ensures the target split maintains the exact class proportions.
    # We use a fixed seed (42) for reproducibility.
    split_dataset = full_dataset.train_test_split(
        test_size=actual_size, 
        stratify_by_column="label", 
        seed=42
    )
    
    # The 'test' split here is actually our target training subset of 'actual_size'
    target_subset = split_dataset["test"]
    
    # Verify label distribution matches the paper's methodology
    labels = [item["label"] for item in target_subset]
    print(f"Label distribution for {actual_size} samples:", dict(Counter(labels)))
    
    # Save in the exact format train.py expects (load_from_disk)
    target_subset.save_to_disk(subset_dir)
    print(f"✅ Successfully saved to {subset_dir}")

print("\n🎉 All subsets generated! You can now run train.py")