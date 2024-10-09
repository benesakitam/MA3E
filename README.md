## Pytorch implementation of [MA3E](https://arxiv.org/abs/2408.01946)

![image](https://github.com/benesakitam/MA3E/blob/main/figs/pipeline.jpg)

## 1. Pre-training Dataset Preparation
Download the MillionAID dataset. The data structure should be as follows:
```
├── MillionAID
│   ├── train
│   │   ├── agriculture
│   │   │   ├── arable_land
│   │   │   │   ├── dry_field
│   │   │   │   │   ├── P0011232.jpg
│   │   │   │   │   ├── ...
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   ├── ...
│   ├── test
│   │   ├── xxx.jpg
│   │   ├── xxx.jpg
│   │   ├── ...
│   ├── ...
```
Since the testing set of the MillionAID dataset does not release labels, to generate random pseudo-labels for pre-training
```
python create_pretraining_labels.py
```
- Please modify the dataset path in `line 6 and 7`

## 2. Pre-training MA3E

To pre-train ViT-Base with **multi-gpu distributed training**, run the following on 1 nodes with 8 GPUs each:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
    --batch_size 128 \
    --model ma3e_vit_base_patch16 \
    --norm_pix_loss \
    --rcrop \
    --crop_size 96 --nums_crop 1 --r_range 45 \
    --b_ratio 0.75 --r_ratio 0.75 \
    --epochs 800 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --dataset millionaid
    --data_path ${Your Million-AID_DIR}
```
- Here the effective batch size is 128 (`batch_size` per gpu) * 1 (`nodes`) * 8 (gpus per node) = 1024. If memory or # gpus is limited, use `--accum_iter` to maintain the effective batch size, which is `batch_size` (per gpu) * `nodes` * 8 (gpus per node) * `accum_iter`.

- This repo is based on [`timm==0.3.2`](https://github.com/rwightman/pytorch-image-models), for which a [fix](https://github.com/rwightman/pytorch-image-models/issues/420#issuecomment-776459842) is needed to work with PyTorch 1.8.1+.

## 3. Scene Classification Dataset Preparation

Download scene classification datasets: NWPU-RESISC45, AID, and UC Merced. The data structures should be as follows:

``` data structure
├── NWPU-RESISC45
│   ├── airplane
│   │   ├── airplane_001.jpg
│   │   ├── airplane_002.jpg
│   │   ├── ...
│   ├── airport
│   │   ├── airport_001.jpg
│   │   ├── airport_002.jpg
│   │   ├── ...
│   ├── ...

├── AID
│   ├── Airport
│   │   ├── airport_1.jpg
│   │   ├── airport_2.jpg
│   │   ├── ...
│   ├── BareLand
│   │   ├── bareland_1.jpg
│   │   ├── bareland_1.jpg
│   │   ├── ...
│   ├── ...

├── UC Merced
│   ├── agricultural
│   │   ├── agricultural00.tif
│   │   ├── agricultural01.tif
│   │   ├── ...
│   ├── airplane
│   │   ├── airplane00.tif
│   │   ├── airplane01.tif
│   │   ├── ...
│   ├── ...
```
-For NWPU-RESISC45, randomly sample 20% of the images for each class as the training set and the remaining 80% as the testing set. 

-For AID and UC Merced, these two ratios is all 50%.

## 4. Fine-tuning

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --batch_size 64 \
    --model vit_base_patch16 \
    --finetune ${MA3E_PRETRAIN_CHKPT} \
    --epochs 200 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${Your DIR of NWPU-RESISC45 (2:8) or AID (5:5) or UC Merced (5:5) Dataset}
```

## 5. Linear probing

```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
    --batch_size 256 \
    --model vit_base_patch16 --cls_token \
    --finetune ${MA3E_PRETRAIN_CHKPT} \
    --epochs 100 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${Your DIR of NWPU-RESISC45 (2:8) or AID (5:5) or UC Merced (5:5) Dataset}
```

## 6. Other downstream tasks

For Rotated Object Detection and Semantic Segmentation. Please refer to [Wang et al. repo](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA).

## Visualization of reconstruction

![image](https://github.com/benesakitam/MA3E/blob/main/figs/vis.jpg)

## Acknowledgments
This repo is a modification on the [MAE repo](https://github.com/facebookresearch/mae). Installation and preparation follow that repo.

## Citation
Please cite this paper if it helps your research:
```
@article{li2024masked,
  title={Masked angle-aware autoencoder for remote sensing images},
  author={Li, Zhihao and Hou, Biao and Ma, Siteng and Wu, Zitong and Guo, Xianpeng and Ren, Bo and Jiao, Licheng},
  journal={arXiv preprint arXiv:2408.01946},
  year={2024}
}
```
