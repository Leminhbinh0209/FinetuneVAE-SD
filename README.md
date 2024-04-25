This repository is created to fine-tune your VAE model of Stable Diffusion model, which you can change input image size, or with a new dataset.

Note: I follow the guidance [here](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/README.md), in which some first epochs are trained with (l1 + Lpips), later epochs are trained with (l2 + 0.1*Lpips) loss.

Please download pre-trained SD model [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt), and put into `sd_model` folder.

Copy `ldm` folder from this [repo](https://github.com/lllyasviel/ControlNet/tree/main?tab=readme-ov-file) into your current directory.


For training, run the following script:

```python 
python train.py \
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
TO-DO: As the denoisor of SD is not fine-tuned in this code, training the VAE alone on new dataset may shift the latent space. There may need to regularize the old the current latent vector with old latent vector in the teach-student: `reg = (self.model.encdoe(x).saples() - old_vae.encode(x).sample().detach()).pow(2)`
