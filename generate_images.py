#
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import os
import argparse
from area import rescale_area
from denoiser import get_denoiser

import random


def fft2_area(img, siz):
    img = np.fft.fft2(img, axes=(0, 1), norm='ortho')
    img_energy = np.abs(img)**2
    img_energy = rescale_area(rescale_area(img_energy, siz, 0), siz, 1)
    img_energy = np.fft.fftshift(img_energy, axes=(0, 1))
    return img_energy


def imread(filename):
    return np.asarray(Image.open(filename).convert('RGB'))/256.0


def extraction(files_path, output_dir, output_code):

    print("Starting the generation of the fingerprints")
    print("Path of the images " + files_path)
    filenames = glob.glob(files_path + "/*")
    random.shuffle(filenames)

    fund = get_denoiser(1, True)
    siz = 222
    filenames = filenames[:1000]
    res_fft2 = [fft2_area(fund(imread(_)), siz) for _ in tqdm(filenames)]
    res_fft2_mean = np.mean(res_fft2, 0)
    res_fcorr_mean = np.fft.ifftshift(np.real(np.fft.ifft2(
        np.fft.ifftshift(res_fft2_mean, axes=(0, 1)), axes=(0, 1))), axes=(0, 1))

    dist_out = dict()
    dist_out['res_fft2_mean'] = res_fft2_mean
    dist_out['res_fcorr_mean'] = res_fcorr_mean

    # saving figures

    center_x = (res_fcorr_mean.shape[1]+1)//2
    center_y = (res_fcorr_mean.shape[0]+1)//2
    extent = [-center_x-1, res_fcorr_mean.shape[1]-center_x,
              res_fcorr_mean.shape[0]-center_y, -center_y-1]  # (left, right, bottom, top)

    energy2 = np.mean(res_fft2_mean)
    res_fcorr_mean = res_fcorr_mean * 256 / 4 / energy2
    res_fft2_mean = res_fft2_mean / 4 / energy2
    figures_output_dir = os.path.join(output_dir, output_code)
    os.makedirs(figures_output_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    plt.imshow((np.mean(res_fft2_mean, -1)).clip(0, 1),
               clim=[0, 1], extent=[-0.5, 0.5, 0.5, -0.5])
    plt.xticks([])
    plt.yticks([])
    fig.savefig(figures_output_dir+'/fft2_gray.png',
                bbox_inches='tight', pad_inches=0.0)

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(np.mean(res_fcorr_mean, -1).clip(-0.5, 0.5),
               clim=[-0.5, 0.5], extent=extent)
    plt.xlim(-32, 32)
    plt.ylim(-32, 32)
    plt.xticks([])
    plt.yticks([])
    fig.savefig(figures_output_dir+'/acor_gray.png',
                bbox_inches='tight', pad_inches=0.0)
    np.savez(figures_output_dir+'/data.npz', **dist_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", type=str,
                        help="The path where the images are stored")
    parser.add_argument("--out_dir", type=str,
                        help="The path where to save the images")
    parser.add_argument("--out_name", type=str,
                        help="The name of the folder in which to save the images and the numpy arrays")
    args = vars(parser.parse_args())
    extraction(args['files_path'], args['out_dir'], args['out_name'])
