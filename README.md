# medlora-3d — LoRA vs Full-FT for 3D Medical Segmentation

**Task:** quantify the crossover behavior of LoRA vs Full Fine-Tuning on 3D segmentation (MSD datasets), using MONAI Swin-UNETR with CT-SSL encoder init.

## Quickstart (Colab)
```bash
pip install -r requirements.txt
# FIRST RUN ONLY: this downloads MSD tasks on-demand inside the first run
python main.py \
  --model swinv1 \
  --dataset Task03_Liver \
  --method lora \
  --train-fraction 20 \
  --seed 0 \
  --epochs 100 --early-stopping true --patience 10 --min-epochs 30
  ```
