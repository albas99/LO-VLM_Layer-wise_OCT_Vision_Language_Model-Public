"""
model.py
--------
BLIP model wrapper for fine-tuning on OCT image captioning.
"""

import os
import json
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, BlipForConditionalGeneration
from torchmetrics.text.bert import BERTScore
from torchmetrics.text import BLEUScore, ROUGEScore

from dataset import ImageCaptioningDataset
from utils import count_trainable_params

# import mlflow
# import dagshub
# 
# mlflow.set_tracking_uri('https://dagshub.com/albas99/oct-vlm.mlflow')
# dagshub.init(repo_name = 'oct-vlm', repo_owner = 'albas99')
# 
# mlflow.pytorch.autolog()


class ImageCaptioningModel:
    """
    Wrapper for BLIP training and evaluation.
    """

    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device="cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Print number of trainable params
        print(f"Number of trainable parameters: {count_trainable_params(self.model)}")
        

    # --------------------------------------------------------
    def train(self, train_dataset, val_dataset, imgs_dir,
        save_dir,num_epochs=50, batch_size=32, validation_steps=20):
        """
        Training loop for BLIP model.
        """
        # DataLoaders
        # added `imgs_dir` remove later
        # after transforming data to std hf format
        train_dataloader = DataLoader(
            ImageCaptioningDataset(train_dataset, self.processor, imgs_dir),
            shuffle=True, batch_size=batch_size, pin_memory=True,
        )
        # here too `imgs_dir`
        val_dataloader = DataLoader(
            ImageCaptioningDataset(val_dataset, imgs_dir, self.processor),
            batch_size=4, pin_memory=True,
        )

        # Optimizer + Scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        # Logging + checkpoint dirs
        log_dir = os.path.join(save_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        step_counter = 0

        # -------------------------------
        # Epoch loop
        # -------------------------------
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_dataloader, desc="Training")

            for _, batch in enumerate(progress_bar):
                input_ids = batch.pop("input_ids").to(self.device)
                pixel_values = batch.pop("pixel_values").to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids,
                                     pixel_values=pixel_values,
                                     labels=input_ids)
                loss = outputs.loss
                loss = loss.mean()
                total_loss += loss.item()

                # Backprop
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                step_counter += 1

                progress_bar.set_postfix({"Loss": loss.item()})

                # Logging
                avg_train_loss = total_loss / len(train_dataloader)
                writer.add_scalar("Loss/Train", avg_train_loss, step_counter)
                writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], step_counter)

                # Scheduler step
                scheduler.step()

            # Validation at epoch end
            self.evaluate(val_dataloader, step_counter, writer, checkpoint_dir)

        writer.close()

    # --------------------------------------------------------
    def evaluate(self, val_dataloader,imgs_dir, step_counter, writer, checkpoint_dir): #added `imgs_dir` remove if not working
        """
        Validation loop + checkpoint saving.
        """
        self.model.eval()
        val_loss = 0
        val_results = []

        total_val_samples = len(val_dataloader.dataset)

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                image_label = batch.pop("image_label")
                original_text = batch.pop("original_text")
                pixel_values = batch["pixel_values"].to(self.device)

                # Forward pass
                outputs = self.model(input_ids=input_ids,
                                     pixel_values=pixel_values,
                                     labels=input_ids)
                val_loss += outputs.loss.item()

                # Generate captions
                base_model = self.model.module if hasattr(self.model, 'module') else self.model
                generated_ids = base_model.generate(pixel_values=pixel_values, max_length=256)
                generated_text = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )

                for i in range(len(batch["input_ids"])):
                    val_results.append({
                        "image_label": image_label[i],
                        "generated_text": generated_text[i],
                        "Expected_Text": original_text[i],
                    })

        avg_val_loss = val_loss / total_val_samples
        writer.add_scalar("Loss/Val", avg_val_loss, step_counter)

        # Save results
        results_path = os.path.join(checkpoint_dir, f"val_results_step_{step_counter}.json")
        with open(results_path, "w") as f:
            json.dump(val_results, f, indent=4)

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{step_counter}.pth")
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(base_model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
