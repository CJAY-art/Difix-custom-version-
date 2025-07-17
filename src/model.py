import os
import torch
import requests
from tqdm import tqdm
from typing import Optional,Tuple,Dict,Any
from diffusers import DDPMScheduler


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("/pub/data/cjl/difix/scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    sample = self.conv_in(sample)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
    # middle
    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample

def my_attn_down_forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, ...]]:
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        lora_scale = cross_attention_kwargs.get("scale", 1.0)

        output_states = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            cross_attention_kwargs.update({"scale": lora_scale})
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)
            batch_view, channel, height, width = hidden_states.shape
            ## ( B V ) H W C ---> B ( V H W ) C
            hidden_states=hidden_states.view(batch_view,channel,-1).transpose(1,2).contiguous().view(batch_view//2,-1,channel)
            hidden_states=hidden_states.transpose(1,2).contiguous().unsqueeze(2)
            hidden_states = attn(hidden_states, **cross_attention_kwargs)
            ## B ( V H W ) C ---> ( B V ) H W C
            hidden_states=hidden_states.squeeze(2).transpose(1,2).contiguous().view(batch_view,-1,channel)
            hidden_states=hidden_states.transpose(1,2).contiguous().view(batch_view,channel,height,width)
            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                if self.downsample_type == "resnet":
                    hidden_states = downsampler(hidden_states, temb=temb, scale=lora_scale)
                else:
                    hidden_states = downsampler(hidden_states, scale=lora_scale)

         
            output_states += (hidden_states,)

        return hidden_states, output_states


def my_attn_up_forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, temb, scale=scale)
            cross_attention_kwargs = {"scale": scale}
            batch_view, channel, height, width = hidden_states.shape

            hidden_states=hidden_states.view(batch_view,channel,-1).transpose(1,2).contiguous().view(batch_view//2,-1,channel)
            hidden_states=hidden_states.transpose(1,2).contiguous().unsqueeze(2)
            hidden_states = attn(hidden_states, **cross_attention_kwargs)
            hidden_states=hidden_states.squeeze(2).transpose(1,2).contiguous().view(batch_view,-1,channel)
            hidden_states=hidden_states.transpose(1,2).contiguous().view(batch_view,channel,height,width)


        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                if self.upsample_type == "resnet":
                    hidden_states = upsampler(hidden_states, temb=temb, scale=scale)
                else:
                    hidden_states = upsampler(hidden_states, scale=scale)

        return hidden_states



def my_unet_mid_forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                batch_view, channel, height, width = hidden_states.shape

                hidden_states=hidden_states.view(batch_view,channel,-1).transpose(1,2).contiguous().view(batch_view//2,-1,channel)
                hidden_states=hidden_states.transpose(1,2).contiguous().unsqueeze(2)
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states=hidden_states.squeeze(2).transpose(1,2).contiguous().view(batch_view,-1,channel)
                hidden_states=hidden_states.transpose(1,2).contiguous().view(batch_view,channel,height,width)

            hidden_states = resnet(hidden_states, temb)

        return hidden_states


def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")
