import os
import requests
import sys
import copy
from tqdm import tqdm
import torch
# from transformers import AutoTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DModel
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from peft import LoraConfig,get_peft_model
p = "src/"
sys.path.append(p)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from model import make_1step_sched, my_vae_encoder_fwd, my_vae_decoder_fwd,my_attn_up_forward,my_attn_down_forward,my_unet_mid_forward


class TwinConv(torch.nn.Module):
    def __init__(self, convin_pretrained, convin_curr):
        super(TwinConv, self).__init__()
        self.conv_in_pretrained = copy.deepcopy(convin_pretrained)
        self.conv_in_curr = copy.deepcopy(convin_curr)
        self.r = None

    def forward(self, x):
        x1 = self.conv_in_pretrained(x).detach()
        x2 = self.conv_in_curr(x)
        return x1 * (1 - self.r) + x2 * (self.r)


class Difix(torch.nn.Module):
    def __init__(self, pretrained_name=None, pretrained_path=None, ckpt_folder="checkpoints", lora_rank_vae=4):
        super().__init__()
        # self.tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer")
        # self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()
        self.sched = make_1step_sched()

        vae = AutoencoderKL.from_pretrained("/pub/data/cjl/difix/vae")

        vae.encoder.forward = my_vae_encoder_fwd.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = my_vae_decoder_fwd.__get__(vae.decoder, vae.decoder.__class__)
        unet = UNet2DModel(in_channels=4,out_channels=4)
        for block in unet.down_blocks:
            if hasattr(block, 'attentions'):
                block.forward = my_attn_down_forward.__get__(block, block.__class__)
        for block in unet.up_blocks:
            if hasattr(block, 'attentions'):
                block.forward = my_attn_up_forward.__get__(block, block.__class__)
        unet.mid_block.forward = my_unet_mid_forward.__get__(unet.mid_block, unet.mid_block.__class__)

        # add the skip connection convs
        vae.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False).cuda()
        vae.decoder.ignore_skip = False
        
    
        print("Zero skip conv")
        torch.nn.init.constant_(vae.decoder.skip_conv_1.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_2.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_3.weight, 1e-5)
        torch.nn.init.constant_(vae.decoder.skip_conv_4.weight, 1e-5)

        target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        vae_lora_config = LoraConfig(r=lora_rank_vae, init_lora_weights="gaussian",
            target_modules=target_modules_vae)
        
        freeze_encoder=vae.encoder
        vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        vae.encoder=freeze_encoder

        self.lora_rank_vae = lora_rank_vae
        self.target_modules_vae = target_modules_vae

        unet.enable_xformers_memory_efficient_attention()
        unet.to("cuda")
        vae.to("cuda")
        self.unet, self.vae = unet, vae
        self.vae.decoder.gamma = 1
        self.timesteps = torch.tensor([99], device="cuda").long()

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.decoder.train()
        for n, _p in self.unet.named_parameters():
            _p.requires_grad = True
        for n, _p in self.vae.encoder.named_parameters():
            _p.requires_grad = False
        for n, _p in self.vae.decoder.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
            else :
                _p.requires_grad = False
        self.vae.decoder.skip_conv_1.requires_grad_(True)
        self.vae.decoder.skip_conv_2.requires_grad_(True)
        self.vae.decoder.skip_conv_3.requires_grad_(True)
        self.vae.decoder.skip_conv_4.requires_grad_(True)

    def forward(self, c_t,deterministic=True, r=1.0, noise_map=None):
        if deterministic:
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            model_pred = self.unet(encoded_control, self.timesteps,).sample
            x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
            x_denoised = x_denoised.to(model_pred.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        else:
            # scale the lora weights based on the r value
            self.unet.set_adapters(["default"], weights=[r])
            set_weights_and_activate_adapters(self.vae, ["vae_skip"], [r])
            encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
            # combine the input and noise
            unet_input = encoded_control * r + noise_map * (1 - r)
            self.unet.conv_in.r = r
            unet_output = self.unet(unet_input, self.timesteps).sample
            self.unet.conv_in.r = None
            x_denoised = self.sched.step(unet_output, self.timesteps, unet_input, return_dict=True).prev_sample
            x_denoised = x_denoised.to(unet_output.dtype)
            self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
            self.vae.decoder.gamma = r
            output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)
        return output_image

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_target_modules"] = self.target_modules_vae
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k or "skip" in k}
        torch.save(sd, outf)

    def load_model(self, model_path, device=None):
        """
        加载保存的模型权重
        Args:
            model_path: 模型权重文件路径
            device: 目标设备 (None表示自动选择)
        Returns:
            dict: 包含加载信息的字典
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 加载保存的权重字典
        checkpoint = torch.load(model_path, map_location=device)
        
        # 2. 验证必需参数
        required_keys = [
            "vae_lora_target_modules",
            "rank_vae",
            "state_dict_unet",
            "state_dict_vae"
        ]
        for key in required_keys:
            if key not in checkpoint:
                raise KeyError(f"Missing required key in checkpoint: {key}")

        # 3. 重建VAE LoRA配置
        self.target_modules_vae = checkpoint["vae_lora_target_modules"]
        self.lora_rank_vae = checkpoint["rank_vae"]
        
        # 4. 重新初始化VAE适配器（确保结构一致）
        vae_lora_config = LoraConfig(
            r=self.lora_rank_vae,
            init_lora_weights="gaussian",
            target_modules=self.target_modules_vae
        )

        # self.vae.add_adapter(vae_lora_config, adapter_name="vae_skip")
        
        # 5. 加载UNet权重（只更新lora和conv_in相关参数）
        unet_state_dict = checkpoint["state_dict_unet"]
        current_unet_sd = self.unet.state_dict()
        
        unet_updates = {}
        for k, v in unet_state_dict.items():
            if k in current_unet_sd:
                if v.shape != current_unet_sd[k].shape:
                    print(f"Warning: Shape mismatch for UNet parameter {k}, "
                        f"expected {current_unet_sd[k].shape}, got {v.shape}")
                    continue
                unet_updates[k] = v
            else:
                print(f"Warning: UNet parameter {k} not found in current model")
        
        if unet_updates:
            current_unet_sd.update(unet_updates)
            # 特别处理TwinConv的conv_in_curr权重
            if 'conv_in.weight' in unet_updates and hasattr(self.unet, 'conv_in'):
                if isinstance(self.unet.conv_in, TwinConv):
                    self.unet.conv_in.conv_in_curr.weight.data.copy_(unet_updates['conv_in.weight'])
                    if 'conv_in.bias' in unet_updates:
                        self.unet.conv_in.conv_in_curr.bias.data.copy_(unet_updates['conv_in.bias'])
                else:
                    self.unet.conv_in.load_state_dict(
                        {k.split('.')[-1]: v for k, v in unet_updates.items() if k.startswith('conv_in')},
                        strict=False
                    )
            self.unet.load_state_dict(current_unet_sd, strict=False)
        
        # 6. 加载VAE权重（只更新lora和skip相关参数）
        vae_state_dict = checkpoint["state_dict_vae"]
        current_vae_sd = self.vae.state_dict()
        
        vae_updates = {}
        for k, v in vae_state_dict.items():
            if k in current_vae_sd:
                if v.shape != current_vae_sd[k].shape:
                    print(f"Warning: Shape mismatch for VAE parameter {k}, "
                        f"expected {current_vae_sd[k].shape}, got {v.shape}")
                    continue
                vae_updates[k] = v
            else:
                print(f"Warning: VAE parameter {k} not found in current model")
        
        # 特别处理skip_conv权重初始化
        skip_convs = ['skip_conv_1', 'skip_conv_2', 'skip_conv_3', 'skip_conv_4']
        for conv_name in skip_convs:
            full_key = f"decoder.{conv_name}.weight"
            if full_key in vae_updates:
                getattr(self.vae.decoder, conv_name).weight.data.copy_(vae_updates[full_key])
        
        if vae_updates:
            current_vae_sd.update(vae_updates)
            self.vae.load_state_dict(current_vae_sd, strict=False)
        
        # 7. 返回加载信息
        load_info = {
            'unet_loaded': len(unet_updates),
            'unet_total': len(unet_state_dict),
            'vae_loaded': len(vae_updates),
            'vae_total': len(vae_state_dict),
            'device': str(device)
        }
        
        print(f"Model loaded from {model_path}")
        print(f"UNet: loaded {load_info['unet_loaded']}/{load_info['unet_total']} parameters")
        print(f"VAE: loaded {load_info['vae_loaded']}/{load_info['vae_total']} parameters")
        
        # 确保模型在正确模式
        self.set_eval()

if __name__ =='__main__':
    model=Difix().eval().to("cuda")
    input=torch.randn(2,3,512,512).to("cuda")
    out=model(input)
    print(out.shape)