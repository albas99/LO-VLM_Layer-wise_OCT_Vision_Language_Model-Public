"""
inference.py
------------
Run inference with a fine-tuned BLIP model on OCT datasets.

This script loads a saved model checkpoint, applies it to a test dataset,
and outputs generated captions alongside ground-truth labels and descriptions.
"""

import os
import json
import argparse
import torch
from PIL import Image
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, BlipForConditionalGeneration


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Inference with BLIP on OCT datasets")

    # Data arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="QIAIUNCC/OCT-summary-Dataset",
        help="Dataset name or path (Hugging Face format)",
    )
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to run inference on (default: test)",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="Base BLIP model name from Hugging Face",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to fine-tuned checkpoint (.pth file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference (cuda or cpu)",
    )

    # Output arguments
    parser.add_argument(
        "--output_json",
        type=str,
        default="inference_results.json",
        help="Path to save results as JSON",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default="inference_results.txt",
        help="Path to save human-readable results",
    )

    return parser.parse_args()


def load_model(model_name, checkpoint_path, device):
    """
    Load BLIP model with checkpoint weights.
    """
    # Load base model + processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    full_sd = ckpt.get("model", ckpt)  # some checkpoints save under 'model'

    model_sd = model.state_dict()
    filtered = {
        k: v for k, v in full_sd.items()
        if (k in model_sd) and (v.shape == model_sd[k].shape)
    }

    model.load_state_dict(filtered, strict=False)
    model.eval().to(device)

    return model, processor


def run_inference(model, processor, dataset, device, output_json, output_txt):
    """
    Run inference and save results.
    """
    results = []
    label_names = dataset.features["label"]
    imgs_dir = 'dataset_500/images'
    
    print(label_names)

    for idx, item in enumerate(dataset):
        print(f"Processing sample {idx + 1}/{len(dataset)}")

        # image_id =  f'image_{idx}'

        # Load image + metadata
        image = Image.open(os.path.join(imgs_dir, item["img_path"]))
        label = item["label"]
        # dec_label = label_names[label]
        caption = item["desc"]

        # Preprocess image
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values=pixel_values, max_length=256)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Store result
        results.append({
            "image_path": item["img_path"],
            # "source": item['source'],
            "image_label": label,
            # "image_label_str": dec_label,
            "generated_text": generated_text,
            "Expected_Text": caption,
        })

    # Save JSON
    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)

    # Save plain text for quick viewing
    with open(output_txt, "w", encoding="utf-8") as f:
        for res in results:
            f.write(f"Image: {res['image_path']}\n")
            f.write(f"Generated text: {res['generated_text']}\n")
            f.write("-" * 28 + "\n")

    print(f"\n✅ Inference completed. Results saved to:\n- {output_json}\n- {output_txt}")


def main():
    args = parse_args()

    # Load dataset
    # dataset = load_dataset(args.dataset_name, split=args.dataset_split)
    dataset = load_from_disk(args.dataset_name)

    # Load model
    model, processor = load_model(args.model_name, args.checkpoint_path, args.device)

    # Run inference
    run_inference(model, processor, dataset, args.device,
                  args.output_json, args.output_txt)


if __name__ == "__main__":
    main()
