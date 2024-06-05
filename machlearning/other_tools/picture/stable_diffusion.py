
import os
import torch
from diffusers import StableDiffusionPipeline
os.environ['HF_HOME'] = 'data/huggingface/hub'

def stableDiffusion():
    # 加载预训练的Stable Diffusion模型
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.to(device)

    # 生成图像
    prompt = "A futuristic cityscape"
    image = pipe(prompt).images[0]

    # 保存生成的图像
    image.save("data/stable_diffusion.png")

if __name__=='__main__':
    stableDiffusion()

