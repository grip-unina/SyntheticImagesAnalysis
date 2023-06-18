import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import os
import argparse
import cv2
import random


def imread(filename):
    return np.asarray(Image.open(filename).convert('RGB'))/256.0


def rescale_img(img, siz):
    h, w = img.shape[:2]
    m = min(w, h)
    if m != siz:
        dim = (siz*w//m, siz*h//m)

        # resize image
        if siz < m:
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        else:
            img = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)

        h, w = img.shape[:2]

    assert min(w, h) == siz
    py = (h - siz)//2
    px = (w - siz)//2
    return img[py:(py+siz), px:(px+siz)]


def get_fft2(x):
    x = np.float64(x)
    x = x - np.mean(x, (-3, -2, -1), keepdims=True)
    x = x/np.sqrt(np.mean(np.abs(x**2), (-3, -2, -1), keepdims=True))

    x = np.fft.fft2(x, axes=(-3, -2), norm='ortho')
    x = np.abs(x)**2

    return x


def get_spectrum(power_spec, q_step=None):
    power_spec = np.mean(power_spec, -1)
    power_spec = power_spec / power_spec.size
    H, W = power_spec.shape
    h, w = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')
    r = np.sqrt(h**2 + w**2)
    if q_step is None:
        q_step = 1.0/min(H, W)

    r_quant = np.round(r/q_step)
    freq = np.sort(np.unique(r_quant))
    y = np.asarray([np.sum(power_spec[r_quant == f]) for f in freq])

    return y, q_step*freq


def get_spectrum_angular(power_spec, num = 16):
    power_spec = np.mean(power_spec, -1)
    power_spec = power_spec / power_spec.size
    H, W = power_spec.shape
    h, w = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing='ij')
    r = np.sqrt(h**2 + w**2)
    
    angular = np.round(num * np.arctan2(h, w) / np.pi) % num
    ang_freq = np.sort(np.unique(angular))
    
    y = np.asarray([np.sum(power_spec[(angular==f) & (r>0.1)]) for f in ang_freq])
    
    return y, ang_freq/num


def get_spectra(files_path, output_dir, output_code):

    print("Starting generation of spectra")
    print(files_path)
    filenames = glob.glob(files_path + "/*")
    print(len(filenames))
    random.shuffle(filenames)

    siz = 256
    filenames = filenames[:1000]
    print("Starting to generate fingerprints")
    img_fft2 = [get_fft2(rescale_img(imread(_), siz)) for _ in tqdm(filenames)]

    freq = get_spectrum(img_fft2[0])[1]
    ang_freq = np.pi*get_spectrum_angular(img_fft2[0])[1]

    spectra = [get_spectrum(_)[0] for _ in tqdm(img_fft2)]
    ang_spectra = [get_spectrum_angular(_)[0] for _ in tqdm(img_fft2)]

    spectra_mean = np.mean(spectra, 0)
    ang_spectra_mean = np.mean(ang_spectra, 0)
    spectra_var = np.var(spectra, 0)
    ang_spectra_var = np.var(ang_spectra, 0)

    figures_output_dir = os.path.join(output_dir, output_code)
    os.makedirs(figures_output_dir, exist_ok=True)
    dict_out = dict()
    dict_out['freq'] = freq
    dict_out['spectra_mean'] = spectra_mean
    dict_out['spectra_var'] = spectra_var
    dict_out['ang_freq'] = ang_freq
    dict_out['ang_spectra_mean'] = ang_spectra_mean
    dict_out['ang_spectra_var'] = ang_spectra_var
    np.savez(figures_output_dir+'/spectra.npz', **dict_out)

    # save figures
    fig = plt.figure(figsize=(6, 5))
    plt.plot(freq, spectra_mean, linewidth=2)
    plt.xlabel('$freq$', fontsize=10)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlim([0.2, 0.5])
    plt.ylim([0.0, 0.0012])
    plt.grid()
    fig.savefig(figures_output_dir+'/spectra.png',
                bbox_inches='tight', pad_inches=0.0)

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'}, figsize=(6, 6))
    ang_spectra_mean = np.concatenate((ang_spectra_mean, ang_spectra_mean, ang_spectra_mean[...,:1]),-1)
    ang_freq = np.concatenate((ang_freq, np.pi+ang_freq, 2*np.pi+ang_freq[:1]),0)
    ax.plot(ang_freq, ang_spectra_mean, linewidth=2)
    ax.set_yticks(ax.get_yticks(), list())
    ax.grid('on')
    fig.savefig(figures_output_dir+'/ang_spectra.png',
                bbox_inches='tight', pad_inches=0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", type=str,
                        help="The path where the images are stored")
    parser.add_argument("--out_dir", type=str,
                        help="The path where to save the images")
    parser.add_argument("--out_name", type=str,
                        help="The name of the folder in which to save the images and the numpy arrays")
    args = vars(parser.parse_args())
    get_spectra(args['files_path'], args['out_dir'], args['out_name'])
