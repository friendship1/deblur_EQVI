# Deblurred EQVI
## Most of the code was taken from here...
 - [EQVI](https://github.com/friendship1/deblur_EQVI/blob/main/.gitignore)
 - [UTI-VFI](https://github.com/yjzhang96/UTI-VFI)
 - [CDVD-TSP](https://github.com/csbhr/CDVD-TSP)


little modification is done by: Jungwan Woo and Donghyeon Kim


![visual_comparison](compare.png)


## Preparation (Same explaination from EQVI)
### Dependencies 


you can find more information on [EQVI github](https://github.com/friendship1/deblur_EQVI/blob/main/.gitignore)
1. Install correlation package
2. Download pretrained models from [EQVI github](https://github.com/friendship1/deblur_EQVI/blob/main/.gitignore)
3. Data preparation from [here](https://competitions.codalab.org/competitions/24584#participate-get-data)



## Quick Testing
`CUDA_VISIBLE_DEVICES=0 python interpolate_EQVI.py configs/config_EQVI_test_gopro.py.py`

## Training
`CUDA_VISIBLE_DEVICES=0,1,2,3 python train_EQVI_lap_l1.py --config configs/config_train_EQVI_VTSR_our_data.py`  

## Evaluation
You can evaluate metric PSNR and SSIM from final output image through metric_calculate.py code

