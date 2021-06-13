import glob
import os
from PIL import Image
# import tensorflow as tf
import numpy as np
import math
import cv2

def get_PSNR_SSIM(output, gt, crop_border=4):
    cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
    cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
    psnr = calc_PSNR(cropped_GT, cropped_output)
    ssim = calc_SSIM(cropped_GT, cropped_output)
    return psnr, ssim

def calc_PSNR(img1, img2):
    '''
    img1 and img2 have range [0, 255]
    '''
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calc_SSIM(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    def ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# def log10(x):
#     numerator = tf.math.log(x)
#     denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
#     return numerator / denominator


# def psnr(im1, im2):
#     img_arr1 = numpy.array(im1).astype('float32')
#     img_arr2 = numpy.array(im2).astype('float32')
#     mse = tf.reduce_mean(tf.math.squared_difference(img_arr1, img_arr2))
#     psnr = tf.constant(255**2, dtype=tf.float32)/mse
#     result = tf.constant(10, dtype=tf.float32)*log10(psnr)
#     with tf.Session():
#         result = result.eval()
#     return result

class Logger():
    def __init__(self):
        self.psnrs = []
        self.ssims = []

    def log(self, psnr, ssim):
        self.psnrs.append(psnr)
        self.ssims.append(ssim)
    
    def print(self):
        avg_psnr = np.average(np.asarray(self.psnrs))
        avg_ssim = np.average(np.asarray(self.ssims))
        print(avg_psnr, avg_ssim)
        return avg_psnr, avg_ssim


if __name__ == "__main__":
    UTI_FLAG = 1
    INTER_FRAMES = 5
    uti_interval = INTER_FRAMES // 2
    TOTAL = 8
    gt_path = "/data0/jwwoo/gopro/test"
    pred_path = "/data0/jwwoo/eqvi/EQVI/outputs/gorpro17"
    # pred_path = "/data1/dh6249/data/CDVD-TSP/infer_results/eqvi_out_53"
    ls = glob.glob(gt_path)
    dirs = os.listdir(gt_path)
    print(dirs)
    logger = Logger()
    for d in dirs:
        print(d)
        gt_glob_path = os.path.join(gt_path, d) + "/*.png"
        pred_glob_path = os.path.join(pred_path, d) + "/*.png"

        gt_ls = glob.glob(gt_glob_path)
        gt_ls.sort()
        pred_ls = glob.glob(pred_glob_path)
        pred_ls.sort()
        print(len(gt_ls))
        print(len(pred_ls))
        # print(gt_ls[0:5], pred_ls[0:5])
        print(gt_ls[0:35])
        for path in pred_ls:
            fnm = os.path.basename(path)
            base = fnm[0:6]
            print(path)
            if "_" in fnm:
                off_idx = int(fnm[7]) + 1
                tar = int(off_idx) * (TOTAL // 4) + 1 + int(base)
                # tar_fnm = str(tar).zfill(6) + ".png"
                img_gt_path = gt_ls[tar - 1]
                if UTI_FLAG:
                    if INTER_FRAMES==5 and off_idx == 1:
                        img_gt_path = gt_ls[tar - 1 + 1]
                    if INTER_FRAMES==5 and off_idx == 3:
                        img_gt_path = gt_ls[tar - 1 - 1]
                img_pred = np.array(Image.open(path))
                # img_gt_path = os.path.join(os.path.join(gt_path, d), tar_fnm)
                img_gt = np.array(Image.open(img_gt_path).resize((640, 360)))
                psnr = calc_PSNR(img_pred, img_gt)
                ssim = calc_SSIM(img_pred, img_gt)
                print("between ", fnm, img_gt_path)
                print(psnr, ssim)
                logger.log(psnr, ssim)
            logger.print()
        print("At directory ", d)
        logger.print()
    print("Final log is ")
    logger.print()        
        

