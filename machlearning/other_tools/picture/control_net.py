
import os
import torch
from diffusers import StableDiffusionPipeline, ControlNetModel
os.environ['HF_HOME'] = 'data/huggingface/hub'

def controlNet_test():
    # 加载预训练的 ControlNet 模型和 Stable Diffusion 模型
    controlnet = ControlNetModel.from_pretrained("CompVis/controlnet-canny")
    pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipeline.to("cuda")

    # 定义控制条件 (例如，边缘图像)
    control_image = "data/OIP.jpg"
    control_condition = controlnet.prepare_condition(control_image)

    # 生成图像
    prompt = "A person in a futuristic outfit"
    image = pipeline(prompt, control=control_condition).images[0]

    # 保存生成的图像
    image.save("controlled_generated_image.png")


if __name__=='__main__':
    r = controlNet_test()