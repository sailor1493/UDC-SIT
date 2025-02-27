import os
import glob
import numpy as np
import rawpy as rp
from PIL import Image
from tqdm import tqdm
import imageio
import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--data-directory",
    help="Directory that contains the data. This directory is expected to contain two subdirectories: GT and input.\nThe test set might not have GT images, in which case only the input directory is needed.",
    default="./training/",
)
parser.add_argument(
    "--result-directory",
    help="Directory where the results will be saved.",
    default="./visualize_sit/",
)
args = parser.parse_args()

dir_sit = args.data_directory
subdir = ["GT", "input"]
res_dir = args.result_directory


def load_npy(filepath):
    img = np.load(filepath)
    img = img / 1023

    return img


def save_4ch_npy_png(tensor_to_save, res_dir, fname, save_type):
    file_basename = os.path.basename(fname).replace(".npy", "")
    if save_type is None:
        postfix = ".png"
    else:
        postfix = "_" + save_type + ".png"
    output_path = os.path.join(res_dir, save_type, file_basename + postfix)

    # if file exists, skip
    if os.path.exists(output_path):
        return

    fn_tonumpy = lambda x: x.to("cpu").detach().numpy().transpose(0, 2, 1)
    source_dir = "./background.dng"
    data = rp.imread(source_dir)
    if not os.path.exists(os.path.join(res_dir, save_type)):
        os.makedirs(os.path.join(res_dir, save_type))
    npy = fn_tonumpy(tensor_to_save)
    npy = npy.squeeze() * 1023

    GR = data.raw_image[0::2, 0::2]
    R = data.raw_image[0::2, 1::2]
    B = data.raw_image[1::2, 0::2]
    GB = data.raw_image[1::2, 1::2]
    GB[:, :] = 0
    B[:, :] = 0
    R[:, :] = 0
    GR[:, :] = 0

    w, h = npy.shape[1:]

    GR[:w, :h] = npy[0][:w][:h]
    R[:w, :h] = npy[1][:w][:h]
    B[:w, :h] = npy[2][:w][:h]
    GB[:w, :h] = npy[3][:w][:h]

    newData = data.postprocess()
    start = (0, 464)
    end = (3584, 3024)
    newData = newData[start[0] : end[0], start[1] : end[1]]

    imageio.imsave(output_path, newData)


for sub in subdir:
    load_dir = os.path.join(dir_sit, sub)
    # check if the directory exists
    if not os.path.exists(load_dir):
        print("Directory does not exist: {}".format(load_dir))
        continue
    search_path = os.path.join(load_dir, "*.npy")
    flist = glob.glob(search_path)
    i = 0
    file_count = len(flist)
    pbar = tqdm(flist, total=file_count)
    for fname in pbar:
        pbar.set_description(fname)
        file_basename = os.path.basename(fname).replace(".npy", "")
        if sub is None:
            postfix = ".png"
        else:
            postfix = "_" + sub + ".png"
        output_path = os.path.join(res_dir, sub, file_basename + postfix)

        # if file exists, skip
        if os.path.exists(output_path):
            continue
        npy_to_save = torch.from_numpy(np.float32(load_npy(fname))).permute(2, 0, 1)
        npy_to_save = torch.clamp(npy_to_save, 0, 1)
        save_4ch_npy_png(npy_to_save, res_dir, fname, sub)
