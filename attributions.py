"""
attributions.py

Extract attributions from model predictions.
"""

import os
import json
import argparse
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, BlipForConditionalGeneration
import captum
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from dataset import ImageCaptioningDataset
from torch.utils.data import DataLoader

from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization as viz
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper

def load_model(model_name, checkpoint_path, device):
    """
    Load BLIP model with checkpoint weights
    """
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    ckpt = torch.load(checkpoint_path, map_location = device, weights_only = False)
    full_sd = ckpt.get("model", ckpt)

    model_sd = model.state_dict()
    filtered = {
        k: v for k, v in full_sd.items()
        if (k in model_sd) and (v.shape == model_sd[k].shape)
    }

    model.load_state_dict(filtered, strict = False)
    model.eval().to(device)

    return model, processor

def prep_dataset(dataset, batch_size, processor, device = 'cuda:1'):
    
    ds_instance = ImageCaptioningDataset(dataset, processor)
    dataloader = DataLoader(ds_instance, batch_size = batch_size, pin_memory = True)

    batch = next(iter(dataloader))
    # print(batch)
    
    return batch

def ig_forward_func(pixel_values, input_ids, model, device = 'cuda:1'):
    pixel_values, input_ids = pixel_values.to(device), input_ids.to(device)
    outputs = model(input_ids = input_ids,
                    pixel_values = pixel_values,
                    labels = input_ids)

    logits = outputs.logits

    input_ids_expanded = input_ids.unsqueeze(-1)

    token_logits = torch.gather(logits, dim = 2, index = input_ids_expanded).squeeze(-1)

    # token_logits = token_logits * attention_mask.float().to(device)

    # num_tokens = attention_mask.sum(dim = 1, keepdim = True)
    # num_tokens = torch.clamp(num_tokens, min = 1.0)
    # target = token_logits.sum(dim = 1, keepdim = False) / num_tokens.squeeze(1).to(device)
    target = token_logits.sum(dim = 1, keepdim = False)

    return target

def run_attr(model, batch, dataset, device):
    # print(batch)
    default_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )
    
    # model = ModelInputWrapper(model)
    ig = IntegratedGradients(ig_forward_func)

    input_ids = batch['input_ids'].to(device)
    pixel_values = batch['pixel_values'].to(device)
    # attention_mask = batch['attention_mask'].to(device)
    

    attrs, delta = ig.attribute(inputs = pixel_values, additional_forward_args = (input_ids, model), internal_batch_size = 32, return_convergence_delta = True)

    attrs_np = attrs.cpu().detach().numpy()
    pixel_values_np = pixel_values.cpu().detach().numpy()
    
    batch_size = pixel_values_np.shape[0]
    os.makedirs('attrs/ig/', exist_ok=True)
    
    for i in range(batch_size):
        single_attr = attrs_np[i]
        single_image = pixel_values_np[i]

        single_attr = single_attr.transpose(1, 2, 0)
        single_image = single_image.transpose(1, 2, 0)

        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        single_image = single_image * std + mean
        single_image = np.clip(single_image, 0, 1)
        
        fig, ax = viz.visualize_image_attr_multiple(
            attr=single_attr,
            original_image=single_image,
            methods=['original_image', 'heat_map'],
            signs=['all', 'absolute_value'],
            use_pyplot=True,
            cmap=default_cmap,
            fig_size=(10, 8),
            show_colorbar=True
        )
        
        img_path = dataset['img_path'][i]
        img_id, img_ext = img_path.split('.')
        
        print(f"\n Saving attributions for {img_id}_{i+1}")
        
        fig.savefig(f"attrs/ig/{img_id}_{i+1}.png", bbox_inches="tight")
        
        plt.close(fig)

def main():
    device = 'cuda:1'
    dataset = load_from_disk('dataset_500/hf_dataset/test')
    
    model, processor = load_model('Salesforce/blip-image-captioning-base', 'run-dataset_500/checkpoints/model_step_2050.pth', 'cuda:1')
    
    batch = prep_dataset(dataset, 16, processor)

    # ig_forward_func(batch['pixel_values'], batch['input_ids'], batch['attention_mask'], model)
    run_attr(model, batch, dataset, 'cuda:1')
    # print(batch['input_ids'])
    # print(model(
    #     input_ids = batch['input_ids'].to(device),
    #     pixel_values = batch['pixel_values'].to(device),
    #     labels = batch['input_ids'].to(device)
    # ))
    

if __name__ == "__main__":
    main()