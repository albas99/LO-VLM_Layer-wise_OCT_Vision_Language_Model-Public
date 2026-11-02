"""
dataset.py
----------
Custom PyTorch dataset for OCT image captioning with BLIP.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class ImageCaptioningDataset(Dataset):
    """
    Dataset wrapper for OCT images + textual descriptions.
    """

    # added `imgs_dir`,
    # todo: transform entire ds_full and ds_500 to standard hf format
    def __init__(self, dataset, processor, imgs_dir):
        self.dataset = dataset
        self.processor = processor
        self.imgs_dir = imgs_dir

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # imgs_dir = self.imgs_dir
        # imgs_dir = 'dataset_500/images'
        # Construct structured caption
        text = (
            f"Diagnosed disease: {item['label']},\n"
            f"Layer Information: {item['desc']}"
        )

        # Encode with BLIP processor
        encoding = self.processor(
            images=Image.open(os.path.join(self.imgs_dir, item["img_path"])),
            text=text,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dim
        encoding = {k: v.squeeze() for k, v in encoding.items()}

        # Keep original text + label for evaluation
        encoding["original_text"] = text
        encoding["image_label"] = item["label"]

        return encoding
