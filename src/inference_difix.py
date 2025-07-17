import os
import torch
import random
from accelerate import Accelerator
from Difix import Difix
from my_utils.training_utils import parse_args_paired_training
import lpips
from torchvision import transforms
from PIL import Image
import numpy as np


checkpoint_path="/pub/data/cjl/difix/exp/checkpoints/model_18501.pkl"
dam_images_dir="/pub/data/cjl/difix/result022/ours_10000/renders"
ref_images_dir="/pub/data/cjl/difix/result022/ours_10000/gt"
output_folder="/pub/data/cjl/difix/result022/ours_10000/out"

def inference(args):
    # 初始化 Accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    if accelerator.is_local_main_process:
        print("开始推理...")

    # 加载模型
    difix = Difix(lora_rank_vae=args.lora_rank_vae)
    difix.set_eval()  # 设置为推理模式

    # 加载预训练权重
    if os.path.exists(checkpoint_path):
        difix.load_model(checkpoint_path)
        if accelerator.is_local_main_process:
            print(f"成功加载模型权重: {checkpoint_path}")
    else:
        if accelerator.is_local_main_process:
            print(f"未找到模型权重文件: {checkpoint_path}")
        return

    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            difix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # 移动模型到设备
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    difix.to(accelerator.device, dtype=weight_dtype)

    # 加载输入图像
    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
        ])
        image = transform(image).unsqueeze(0)
        return image

    # 获取损坏图像和参考图像文件夹中的所有图像

    dam_image_files = sorted([f for f in os.listdir(dam_images_dir) if os.path.isfile(os.path.join(dam_images_dir, f))])
    ref_image_files = sorted([f for f in os.listdir(ref_images_dir) if os.path.isfile(os.path.join(ref_images_dir, f))])
    random.shuffle(ref_image_files)
    if len(dam_image_files) != len(ref_image_files):
        if accelerator.is_local_main_process:
            print("损坏图像和参考图像的数量不匹配，请检查文件夹内容。")
        return

    os.makedirs(output_folder, exist_ok=True)

    for dam_file, ref_file in zip(dam_image_files, ref_image_files):
        dam_image_path = os.path.join(dam_images_dir, dam_file)
        ref_image_path = os.path.join(ref_images_dir, ref_file)

        dam_image = load_image(dam_image_path).to(accelerator.device, dtype=weight_dtype)
        ref_image = load_image(ref_image_path).to(accelerator.device, dtype=weight_dtype)


        T = transforms.Compose([
            transforms.Resize((height, width), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        dam_image = dam_image.resize((512, 512), Image.LANCZOS)
        ref_image = ref_image.resize((512, 512), Image.LANCZOS)
        x = torch.stack([T(dam_image), T(ref_image)], dim=0).unsqueeze(0)

        # 推理
        with torch.no_grad():
            x_tgt_pred = difix(x, deterministic=True)[0].unsqueeze(0)

        # 处理输出图像
        output_image = x_tgt_pred[0].cpu()*0.5+0.5
        output_image = transforms.ToPILImage()(output_image)

        # 保存输出图像
        output_path = os.path.join(output_folder, f"output_{os.path.splitext(dam_file)[0]}.png")
        output_image.save(output_path)
        if accelerator.is_local_main_process:
            print(f"推理结果已保存到: {output_path}")

if __name__ == "__main__":
    args = parse_args_paired_training()
    # 修改推理所需参数
    inference(args)