# python3 train.py   --train_base_path /path/to/balanced_training_subsets   --training_sizes 500 2000 5000   --epochs 30   --batch_size 16   --val_batch_size 4   --val_dataset QIAIUNCC/OCT-Text-Dataset   --val_split test   --device cuda

imgs="./data/dataset_full/images/"
ds_full_train="./data/dataset_full/hf_dataset/train/"
ds_full_val="./data/dataset_full/hf_dataset/val/"

CUDA_VISIBLE_DEVICES=1 python3 train.py --train_base_path "${ds_full_train}" \
--val_dataset "${ds_full_val}" \
--imgs_dir "${imgs}" \
--batch_size 16 --epochs 30
