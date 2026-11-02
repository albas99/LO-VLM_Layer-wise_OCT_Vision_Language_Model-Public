import json
import os
import argparse
import re

import numpy as np
import torch
import torchmetrics
from PIL import Image
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics.text.bert import BERTScore
from transformers import AutoProcessor

def parse_args():

    parser = argparse.ArgumentParser(description = "Calculate Inference Metrics")

    parser.add_argument(
        "inf_results",
        type = str,
        help = "Path to JSON file containing inference results",
    )

    parser.add_argument(
        "--name",
        type = str,
        required = True,
        help = "Name of calculated metrics, ideally a combination of dataset+model&checkpoint(if used)"
    )

    parser.add_argument(
        "--model_name_or_path",
        type = str,
        default = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        help = "Name or path to model to be used for evaulation, necessary for calculating BERT scores, global default is `roberta-large` but we're using `BiomedNLP` here since it's medical data/text"
    )

    parser.add_argument(
        "--output_json",
        type = str,
        required = True,
        help = "Path to save metrics in JSON format"
    )

    return parser.parse_args()

def calc_bert(data, output_json, name, device):
    scores = [name]
    bertscore = BERTScore(model_name_or_path='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', all_layers = True, device = device)

    for sample in data:
        ground_text = sample['Expected_Text']
        generated_text = sample['generated_text']
        image_path = sample['image_path']
        metric_scores = bertscore(generated_text, ground_text)
        scores.append({
            "image_path": image_path,
            "ground_text": ground_text,
            "generated_text": generated_text,
            "bertscores": {
                    "precision": metric_scores['precision'].tolist(),
                    "mean_precision": metric_scores['precision'].mean().tolist(),
                    "recall": metric_scores['recall'].tolist(),
                    "mean_recall": metric_scores['recall'].mean().tolist(),
                    "f1": metric_scores['f1'].tolist(),
                    "mean_f1": metric_scores['f1'].mean().tolist()
                    }
        })

    with open(f'metric_scores/bert_{output_json}', "w") as f:
        json.dump(scores, f, indent = 4)

    print(f"Bert Scores calculated. Results saved to {output_json}")

def calc_bleu(data, output_json, name):
    scores = [name]
    bleu_score = BLEUScore(n_gram = 1)

    for sample in data:
        ground_text = sample['Expected_Text'].lower().replace('\n', '')
        generated_report = sample['generated_report']
        image_path = sample['image_path']
        metric_scores = bleu_score(generated_report, ground_text)
        scores.append({
            "image_path": image_path,
            "diagnosed_diseases": sample['diagnosed_diseases'],
            "ground_text": ground_text,
            "generated_report": generated_report,
            "bleuscores": metric_scores.tolist()
        })

    with open(f'metric_scores/bleu_{output_json}', "w") as f:
        json.dump(scores, f, indent = 4)

    print(f"BLEU Scores calculated. Results saved to {output_json}")

def calc_rouge(data, output_json, name):
    scores = []
    rouge = ROUGEScore()

    scores.append(name)
    for sample in data:
        ground_text = sample['Expected_Text'].lower().replace('\n', '')
        generated_report = sample['generated_report']
        image_path = sample['image_path']
        metric_scores = rouge(generated_report, ground_text)
        scores.append({
            "image_path": image_path,
            "diagnosed_diseases": sample['diagnosed_diseases'],
            "ground_text": ground_text,
            "generated_report": generated_report,
            "rouge_scores": {k: v.tolist() for k, v in metric_scores.items()}
        })
        # for k, v in metric_scores.items():
        #     scores.append({k: v.tolist()})

    with open(f'metric_scores/rouge_{output_json}', "w") as f:
        json.dump(scores, f, indent = 4)

    print(f"ROUGE Scores calculated, Results saved to {output_json}")

def clean_preds(data, name):
    label_names = [
      "Choroidal neovascularization",
      "DRUSEN",
      "Diabetic macular edema",
      "NORMAL"
    ]
    for sample in data:
        gen_text = sample['generated_text']
        pattern = r'^diagnosed disease\s*:\s*(?P<diseases>.*?)(?=, layer information),?\s*layer information\s*:\s*(?P<report>.*)$'
        match = re.search(pattern, gen_text, flags = re.IGNORECASE)

        if match:
            diseases = match.group('diseases')
            report = match.group('report')

        sample['diagnosed_diseases'] = diseases
        sample['generated_report'] = report

        if name == 'kermany':
            sample['diagnosed_diseases'] = label_names[int(diseases)]
                

    return data
    

def main():
    args = parse_args()
    res_file = args.inf_results
    out_file = args.output_json
    name = args.name
    with open(res_file) as f:
        res = json.load(f)
    device = 'cuda:1'

    processor = AutoProcessor.from_pretrained('Salesforce/blip-image-captioning-base')

    ## first run used `roberta-large` while second one used `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`

    clean_res = clean_preds(res, name)
    # calc_bert(res, out_file, name, device)
    calc_bleu(clean_res, out_file, name)
    calc_rouge(clean_res, out_file, name)


if __name__ == "__main__":
    main()