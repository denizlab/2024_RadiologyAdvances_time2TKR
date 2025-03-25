# TWIST Pretraining Procedure

This repository contains scripts for training the **TWIST** model from scratch, as well as a version set up with PyTorch Lightning for easier experimentation and faster iteration.

## Contents

- `TWIST_train.py`: Script for training TWIST from scratch.
- `TWIST_lighting_train.py`: Script for training TWIST using PyTorch Lightning.
- `TWIST_lighting_inference.py`: Script for inference and extracting features with TWIST pre-trained models using PyTorch Lightning.

## Train TWIST with pytorch-lightning setup

### SAG_IW_TSE Training

```shell
python TWIST_lighting_train.py \
    --epochs 500 \
    --batch_size 16 \
    --lam1 0.0 \
    --lam2 1.0 \
    --dim 1024 \
    --hid_dim 4096 \
    --optim 'lars' \
    --lr 0.0001 \
    --data_dir ./Datasets/OAI/SAG_IW_TSE \ # path to dataset directory
    --train_df_path ./Datasets/SAG_IW_TSE/SAG_IW_TSE_train.csv \ # path to train csv
    --val_df_path ./Datasets/SAG_IW_TSE/SAG_IW_TSE_val.csv \ # path to val csv
```

### SAG_3D_DESS Training

```shell

python TWIST_lighting_train.py \
    --epochs 500 \
    --batch_size 16 \
    --lam1 0.0 \
    --lam2 1.0 \
    --dim 1024 \
    --hid_dim 4096 \
    --optim 'lars' \
    --lr 0.0001 \
    --data_dir ./Datasets/OAI/SAG_3D_DESS \ # path to dataset directory
    --train_df_path ./metadata/pretraining/SAG_3D_DESS/SAG_3D_DESS_train.csv \ # path to train csv
    --val_df_path ./metadata/pretraining/SAG_3D_DESS/SAG_3D_DESS_val.csv \ # path to val csv
```

## TWIST Inference with pytorch-lightning setup

### SAG_IW_TSE Inference

```shell
python TWIST_lighting_inference.py \
    --epochs 1 \
    --batch_size 1 \
    --data_dir ./Datasets/OAI/SAG_IW_TSE \ 
    --all_df_path ./metadata/downstream/SAG_IW_TSE/SAG_IW_TSE_downstream_train.csv \ # input data for downstream tasks, train/val/test
    --checkpoint_path ./saved_models/SAG_IW_TSE_v1.ckpt \ # model checkpoint or model path
    --save_folder ./extracted_features/ \ # path to save extracted features
    --save_tensor SAG_IW_TSE_OAI_Features \ # name of tensor to save
```

### SAG_3D_DESS Inference

```shell
python TWIST_lighting_inference.py \
    --epochs 1 \
    --batch_size 1 \
    --data_dir ./Datasets/OAI/SAG_3D_DESS \ 
    --all_df_path ./metadata/downstream/SAG_3D_DESS/SAG_3D_DESS_downstream_train.csv \ # input data for downstream tasks, train/val/test
    --checkpoint_path ./saved_models/SAG_3D_DESS_v1.ckpt \ # model checkpoint or model path
    --save_folder ./extracted_features/ \ # path to save extracted features
    --save_tensor SAG_3D_DESS_Features \ # name of tensor to save
```

Extracted features are saved in tensor in the folder 'extracted_features'.