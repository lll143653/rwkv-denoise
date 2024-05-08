import torch
import numpy as np
from torchvision.io import read_image
from torchvision.utils import save_image
from denoise_model import DiffRWKVModel
import argparse
import time
from datetime import datetime
import os
from val_dataset import SIDD_val
from torch.utils.data import DataLoader
from tqdm import tqdm
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidd", type=str, help="sidd val path")
    parser.add_argument("--img", type=str, help="noise img path")
    parser.add_argument("--weight", required=False,
                        default='weights/rwkv6sidd128.39.34db.pth')
    args = parser.parse_args()
    return args

def calculate_psnr(
        img1: torch.Tensor | np.ndarray,
        img2: torch.Tensor | np.ndarray,
) -> float:
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))

def load_model(args):
    model = DiffRWKVModel(
        img_size=64, patch_size=4, in_channels=3, drop_rate=0., embed_dims=256, depth=7, shift_pixel=1
    ).cuda()
    model.load_state_dict(torch.load(args.weight), strict=True)
    return model

def get_timestamp():
    return datetime.now().strftime("%y%m%d-%H%M%S")

def createDir():
    current_time_str = get_timestamp()
    path=os.path.join('output', current_time_str)
    os.makedirs(path, exist_ok=True)
    return path


def main(args):
    model=load_model(args)
    val_dir=createDir()
    with torch.inference_mode():
        if args.sidd:
            data=DataLoader(SIDD_val({
                "sidd_val_dir":args.sidd,
                "len":1280
            }),
            shuffle=False)
            psnrs=[]
            for index,i in enumerate(tqdm(data)):
                clean,noisy=i['clean'].cuda().float()/255.,i['noisy'].cuda().float()/255.
                denoised=model(noisy)
                psnr=calculate_psnr(clean*255.,denoised*255.)
                psnrs.append(psnr)
                save_image(torch.concat([noisy,denoised]),
                        os.path.join(val_dir,f'{index}.{psnr:.2f}.png'))
            print(f"average psnr: {sum(psnrs)/len(psnrs):.2f}")

        if args.img:
            noisy_img=read_image(args.img).unsqueeze(0)
            noisy_img=noisy_img.cuda().float()/255.
            denoised=model(noisy_img)
            save_image(torch.concat([noisy_img,denoised]),
                        os.path.join(val_dir,f'denoised_{os.path.basename(args.img)}'))
        print(f"denoise reult could be found in {val_dir}")


if __name__ == "__main__":
    main(parseArgs())
