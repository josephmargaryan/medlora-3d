# medlora-3d — LoRA vs Full-FT for 3D Medical Segmentation

**Task:** quantify the crossover behavior of LoRA vs Full Fine-Tuning on 3D segmentation (MSD datasets), using MONAI Swin-UNETR with CT-SSL encoder init.

## Quickstart (Colab)
```bash
%%capture
!pip install --upgrade pip
!pip install monai nibabel
!git clone https://github.com/josephmargaryan/medlora-3d.git
%cd medlora-3d
!pip install -e . --no-deps
  ```
then 
```bash
!python main.py \
  --model swinv1 \
  --dataset Task03_Liver \
  --method fft \
  --train-fraction 20 \
  --seed 0 \
  --epochs 50 \
  --early-stopping true \
  --patience 5 \
  --min-epochs 10 \
  --batch-size 2 \
  --num-workers 2 \
  --save-preds true \
  --pred-splits both \
  --save-nii false \
  --num-workers 2 \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0.0
```
# --model {swinv1|swinv2}
# --method {lora|fft}

# --dataset Task03_Liver
#Task01_BrainTumour
#Task02_Heart
#Task03_Liver
#Task04_Hippocampus
#Task05_Prostate
#Task06_Lung
#Task07_Pancreas
#Task08_HepaticVessel
#Task09_Spleen
#Task10_Colon

# --train-fraction 20 \ # 5, 20, 80
