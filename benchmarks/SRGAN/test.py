import rawpy as rp
import imageio
import os
from argparse import ArgumentParser
import tqdm
import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)

from networks.SRGAN import *
from datasets.dataset_pairs_npy import my_dataset_eval
from util import setup_img_save_function

img_save_function = None

def calculate_lfd_lpd(lq, gt):
    fft_lq = torch.fft.fftn(lq, dim=(-2, -1))
    fft_gt = torch.fft.fftn(gt, dim=(-2, -1))
    
    lfd_sum_vector = torch.pow(torch.abs(fft_lq - fft_gt), 2)
    lfd_value = torch.log(lfd_sum_vector.mean() + 1)
    lpd_sum_vector = torch.pow(torch.abs(torch.angle(fft_lq)-torch.angle(fft_gt)), 2)
    lpd_value = torch.log(lpd_sum_vector.mean() + 1)
    
    return lfd_value, lpd_value

def test(
    model: nn.Module,
    test_loader: DataLoader,
    save_image=False,
    save_dir=None,
    experiment_name=None,
    logger=None,
    data_itself=False,
):
    global img_save_function
    model.eval()
    ddp_loss = torch.zeros(5).to(0)
    
    if data_itself:
        model = lambda x: x

    iter_bar = tqdm.tqdm(test_loader)
    with torch.no_grad(), open(os.path.join(save_dir, "log.csv"), "w") as f:
        f.write("IMG,PSNR,SSIM,LFD,LPD\n")
        for data, target, img_name in iter_bar:
            img_name = img_name[0].split(".")[0]

            data, target = data.to(0), target.to(0)
            output = model(data)
            # if ouptut is list, take only the last one
            if isinstance(output, list):
                output = output[-1]
            psnr = peak_signal_noise_ratio(output, target)
            ddp_loss[0] += psnr
            ssim = structural_similarity_index_measure(output, target)
            ddp_loss[1] += ssim
            ddp_loss[2] += len(data)
            
            lfd, lpd = calculate_lfd_lpd(output, target)
            ddp_loss[3] += lfd
            ddp_loss[4] += lpd 

            if logger:
                msg = f"Img: {img_name}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LFD: {lfd:.4f}, LPD: {lpd:.4f}"
                logger.info(msg)
            f.write(f"{img_name},{psnr:.4f},{ssim:.4f},{lfd:.4f},{lpd:.4f}\n")
            f.flush()

            if save_image and save_dir is not None:
                filename = f"{experiment_name}_{img_name}.png"
                img_save_function(output, os.path.join(save_dir, filename))
    msg = f"Test set: Average PSNR: {ddp_loss[0] / ddp_loss[2]:.4f}, Average SSIM: {ddp_loss[1] / ddp_loss[2]:.4f}"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def main():
    global img_save_function

    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-input", type=str, default="data/test/input")
    parser.add_argument("--test-GT", type=str, default="data/test/GT")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--norm", type=str, required=True, choices=["norm", "tonemap"])
    parser.add_argument("--data-itself", action="store_true")
    args = parser.parse_args()

    print("Creating test dataset...", end=" ", flush=True)
    test_dataset = my_dataset_eval(args.test_input, args.test_GT, args.norm)
    print("Done!", flush=True)

    print("Setup image save function...", end=" ", flush=True)
    img_save_function = setup_img_save_function(args.channels)

    print("Creating model...", end=" ", flush=True)
    model = Generator(io_channels=args.channels).to(0)
    print("Done!", flush=True)

    print("Loading model...", end=" ", flush=True)
    model_path = args.model_path
    loaded = torch.load(model_path)
    to_load = loaded
    model.load_state_dict(to_load, strict=False)
    print("Done!", flush=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"logs/{args.name}_test.log")
    logger.addHandler(handler)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    print("Creating test loader...", end=" ", flush=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        pin_memory_device="cuda:0",
        drop_last=False,
    )
    print("Done!", flush=True)
    save_dir = f"results/{args.name}"
    os.makedirs(save_dir, exist_ok=True)
    test(
        model,
        test_loader,
        save_image=True,
        save_dir=save_dir,
        experiment_name=args.name,
        logger=logger,
        data_itself=args.data_itself,
    )


if __name__ == "__main__":
    main()
