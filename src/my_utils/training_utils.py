import os
import random
import argparse
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
from glob import glob


def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the loss function
   
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_rec", default=1.0, type=float)
    parser.add_argument("--lambda_gram", default=0.5, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", type=str,default="/pub/data/sunhao24")
    parser.add_argument("--image_size", default="512", type=int)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_difix", help="The name of the wandb project to log to.")

    # details about the model architecture
    
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)

    parser.add_argument("--lora_rank_vae", default=16, type=int)

    # training details
    parser.add_argument("--output_dir", default="/pub/data/cjl/difix/exp",type=str)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=15_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=4,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split):

        super().__init__()
        if split == "train":
            self.data_dir=os.path.join(dataset_folder,"trainset")

        elif split == "test":
            self.data_dir=os.path.join(dataset_folder,"testset")
            
        self.pairs = self._find_image_gt_pairs()
        print(split+" image pairs:",len(self.pairs))
        self.T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _find_image_gt_pairs(self):
        pairs = []
        
        for name in os.listdir(self.data_dir):
            dam=os.path.join(self.data_dir,name,"damaged")
            gt=os.path.join(self.data_dir,name,"gt")
            ref=os.path.join(self.data_dir,name,"ref")

            # 遍历img下的所有子文件夹
            for img in os.listdir(dam):
                
                img_path = os.path.join(dam, img)
                
                # 构建对应的GT路径
                
                gt_path = os.path.join(gt, img[-9:])
                ref_path=os.path.join(ref, img[-9:])
                # ref_path =os.path.join(self.ref, img_path)
                
                pairs.append((img_path, gt_path, ref_path))
        
        return pairs



    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):
 
        img_name,gt,ref= self.pairs[idx]
        dam = Image.open(img_name)
        gt=Image.open(gt)
        ref=Image.open(ref)

        input = self.T(dam)
        ref=self.T(ref)
        gt =self.T(gt)

        return {
            "dam": input,
            "gt":gt,
            "ref": ref,
        }

class PairedData_test(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split):

        super().__init__()
        if split == "train":
            self.data_dir=os.path.join(dataset_folder,"trainset")

        elif split == "test":
            self.data_dir=os.path.join(dataset_folder,"testset")
            
        self.pairs = self._find_image_gt_pairs()
        print(split+" image pairs:",len(self.pairs))
        self.T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def _find_image_gt_pairs(self):
        pairs = []
        
        for name in os.listdir(self.data_dir):
            dam=os.path.join(self.data_dir,name,"damaged")
            gt=os.path.join(self.data_dir,name,"gt")
            ref=os.path.join(self.data_dir,name,"ref")

            # 遍历img下的所有子文件夹
            for img in os.listdir(dam):
                
                img_path = os.path.join(dam, img)
                
                # 构建对应的GT路径
                
                gt_path = os.path.join(gt, img[-9:])
                ref_path=os.path.join(ref, img[-9:])
                # ref_path =os.path.join(self.ref, img_path)
                
                pairs.append((img_path, gt_path, ref_path))
        
        return pairs



    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, idx):
 
        img_name,gt,ref= self.pairs[idx]
        dam = Image.open(img_name)
        gt=Image.open(gt)
        ref=Image.open(ref)

        input = self.T(dam)
        ref=self.T(ref)
        gt =self.T(gt)

        return {
            "dam": input,
            "gt":gt,
            "ref": ref,
        }


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class StyleLoss(nn.Module):
    def __init__(self, style_layers=None):
        super(StyleLoss, self).__init__()
        if style_layers is None:
            style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.style_layers = style_layers

        # 加载预训练的VGG19模型
        self.vgg = models.vgg19(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        self.layer_name_mapping = {
                    '3': "relu1_2",
                    '8': "relu2_2",
                    '15': "relu3_3",
                    '22': "relu4_3"
                }
        
        for p in self.parameters():
            p.requires_grad = False

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

        self.vgg.cuda()

    @staticmethod
    def gram_matrix(x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    @staticmethod
    def style_loss(input_tensor, target_tensor):
        input_gram = StyleLoss.gram_matrix(input_tensor)
        target_gram = StyleLoss.gram_matrix(target_tensor)
        loss = nn.MSELoss()(input_gram, target_gram).mean()
        return loss

    def extract_features(self, x):
        outs = []
        x = (x + 1.0) / 2.0
        x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=True)

        outs = []
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                outs.append(x)
        return outs

    def forward(self, x, y):

        x_features = self.extract_features(x)
        y_features = self.extract_features(y)

        style_loss = 0
        for a, b in zip(x_features, y_features):
            style_loss += self.style_loss(a, b)

        return style_loss
    
if __name__ =='__main__':
    x=torch.randn(2,3,256,256).cuda()
    y=torch.randn(2,3,256,256).cuda()
    l=StyleLoss()
    print(l(x,y))