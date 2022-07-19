python train.py \
  --dataset tieredImageNet \
  --backbone resnet34 \
  --lr 0.05 \
  --batch_size 512 \
  --epochs 200 \
  --dim_hidden 2048 \
  --data_path [your imagenet-train-folder] \
  --save_path [your save-folder]