This repository is created to fine-tune your VAE of Stable Diffusion model, which you can change input size.


Please download pre-trained SD model [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt), and put into `sd_model` folder.

Copy `ldm` folder from this [repo](https://github.com/lllyasviel/ControlNet/tree/main?tab=readme-ov-file) into your current directory.


For training, run the following script:

```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python train_vae.py \
						--data_dir <YOUR DATA FOLDER>
						--batch_size 2 \
						--num_epochs 20 \
						--lr 2e-5 \
                        --val_size 0.1 \
						--precision 16 \
						--image_size <YOUR DESIRED SIZE> \
						--lpips_loss_weight 1.0 \
						--ema_decay 0.99\
```