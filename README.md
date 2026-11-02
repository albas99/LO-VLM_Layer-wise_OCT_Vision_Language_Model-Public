---
dataset_info:
  features:
  - name: source
    dtype: string
  - name: label
    dtype: string
  - name: desc
    dtype: string
  - name: image
    dtype: image
  splits:
  - name: train
    num_bytes: 5284064802.5
    num_examples: 26668
  - name: test
    num_bytes: 130120875.0
    num_examples: 664
  download_size: 5404516204
  dataset_size: 5414185677.5
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
