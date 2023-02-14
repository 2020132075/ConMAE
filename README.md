# ConMAE

#This is a PyTorch/GPU implementation of the paper [ConMAE: Contour Guided MAE for Unsupervised Vehicle Re-Identification](CCDC 2023)

This implementation is based on the secondary development of [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) and [Cluster Contrast for Unsupervised Person Re-Identification](https://arxiv.org/abs/2103.11568), thanks to the authors of the above work.

When using this code, you will first pretrain using the code in "\ConMAE\upstream", then train the model obtained from the above step in unsupervised vehicle reidentification using the code in "\ConMAE\downstream".

### Prepare ImageNet Pre-trained Models for MAE
When training with the backbone of [MAE](https://arxiv.org/abs/2111.06377), you need to download the ImageNet-pretrained ViT-Base model from this [link](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) 

### upstream
To pre-train ViT-Base with **main_pretrain.py**, we utilize 2 GTX-2080TI GPUs for training.
``` 
    --height 224 --width 224\
    --batch_size 64 \
    --model mae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --blr 1.5e-4 --weight_decay 0.05 \
```
### downstream
When training the ViT we got in upstream with **main.py**, we utilize 2 GTX-2080TI GPUs.
``` 
    --height 224 --width 224\
    --batch_size 64 \
    --model MAE_base \
    --epochs 60 \
    --iters 400\
    --momentum 0.1\
    --eps 0.7\
    --num-instances 4\
    --lr 1.5e-5\
    --weight_decay 5e-4 \
```
###If you have any questions, please email "2020132075@chd.edu.cn".

