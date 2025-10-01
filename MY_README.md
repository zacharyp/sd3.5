
## stable difussion 3.5 

```

python3 -m venv stablediff3
source stablediff3/bin/activate

pip install -r requirements.txt

mkdir models
cd models

cd ~/code && git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-medium

cd models & ln -s ~/code/stable-diffusion-3.5-medium/sd3.5_medium.safetensors

download https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/clip_l.safetensors

download https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/clip_g.safetensors

download https://huggingface.co/stabilityai/stable-diffusion-3.5-large/blob/main/text_encoders/t5xxl_fp8_e4m3fn.safetensors
cd models & ln -s t5xxl_fp8_e4m3fn.safetensors t5xxl.safetensors

```

