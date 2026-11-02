"""
train.py
--------
Main entry point for fine-tuning BLIP on OCT datasets.
"""

import os
import argparse
from collections import Counter
from datasets import load_from_disk, load_dataset
import torch
import gc

from model import ImageCaptioningModel


def parse_args():
    """
    Parse command-line arguments for training configuration.
    """
    parser = argparse.ArgumentParser(description="BLIP fine-tuning for OCT captioning")

    # Data arguments
    parser.add_argument(
        "--val_dataset",
        type=str,
        default="QIAIUNCC/OCT-summary-Dataset",
        help="Hugging Face dataset name or path for validation set",
    )
    parser.add_argument(
        "--val_split",
        type=str,
        default="test",
        help="Validation dataset split (default: test)",
    )
    parser.add_argument(
        "--train_base_path",
        type=str,
        required=True,
        help="Base path containing pre-saved training subsets",
    )
    # parser.add_argument(
    #     "--training_sizes",
    #     type=int,
    #     nargs="+",
    #     default=[2000, 5000, 10000, 15000, 20000, 26000],
    #     help="List of training subset sizes to use",
    # )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="Hugging Face model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on (cuda or cpu)",
    )
    parser.add_argument(
        "--imgs_dir",
        type=str,
        required=True,
        help="Directory to where images are store"
    )

    # Training arguments
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--val_steps", type=int, default=6, help="Validation frequency in steps")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load validation dataset
    # val_dataset = load_dataset(args.val_dataset, split=args.val_split)
    val_dataset = load_from_disk(args.val_dataset)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Initialize model
    image_captioning_model = ImageCaptioningModel(
        model_name=args.model_name, device=args.device
    )

    # Train on each specified subset size
    # for size in args.training_sizes:
    #     subset_path = os.path.join(args.train_base_path, str(size))
    #     if os.path.exists(subset_path):
    #         train_subset = load_from_disk(subset_path)
    #         print(f"\nTraining with {len(train_subset)} samples")
    #         labels = [item["label"] for item in train_subset]
    #         print(f"Label distribution for {size} samples:", Counter(labels))

    #         save_dir = f"run-base-{size}"
    #         image_captioning_model.train(
    #             train_dataset=train_subset,
    #             val_dataset=val_dataset,
    #             save_dir=save_dir,
    #             num_epochs=args.epochs,
    #             batch_size=args.batch_size,
    #             validation_steps=args.val_steps,
    #         )
    #     else:
    #         print(f"Subset for size {size} not found at {subset_path}")

    # remove `imgs_dir` later
    train_ds = load_from_disk(args.train_base_path)
    print(f"\nTraining with {len(train_ds)}")
    labels = [item['label'] for item in train_ds]
    print("Label distribution:", Counter(labels))
    save_dir = "runs/run-dataset_5000"
    image_captioning_model.train(
        train_dataset = train_ds,
        val_dataset = val_dataset,
        save_dir = save_dir,
        num_epochs = args.epochs,
        batch_size = args.batch_size,
        validation_steps = args.val_steps,
        imgs_dir = args.imgs_dir
    )

if __name__ == "__main__":
    main()
