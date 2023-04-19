import argparse
import glob
import numpy as np
import os
import cv2
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img

from basicsr.utils.registry import ARCH_REGISTRY

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--test_path', type=str, default='datasets/ffhq/ffhq_512')
    parser.add_argument('-o', '--save_root', type=str, default='./results/vqgan_rec')
    parser.add_argument('--codebook_size', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default='./experiments/pretrained_models/vqgan/net_g.pth')
    args = parser.parse_args()

    if args.save_root.endswith('/'):  # solve when path ends with /
        args.save_root = args.save_root[:-1]
    dir_name = os.path.abspath(args.save_root)
    os.makedirs(dir_name, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_path = args.test_path
    save_root = args.save_root
    ckpt_path = args.ckpt_path
    codebook_size = args.codebook_size

    vqgan = ARCH_REGISTRY.get('VQAutoEncoder')(512, 64, [1, 2, 2, 4, 4, 8], 'nearest',
                                                codebook_size=codebook_size).to(device)
    checkpoint = torch.load(ckpt_path)['params_ema']

    vqgan.load_state_dict(checkpoint)
    vqgan.eval()

    for img_path in sorted(glob.glob(os.path.join(test_path, '*.[jp][pn]g'))):
        img_name = os.path.basename(img_path)
        print(img_name)
        img = cv2.imread(img_path)
        img = img2tensor(img / 255., bgr2rgb=True, float32=True)
        normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            output = vqgan(img)[0]
            output = tensor2img(output, min_max=[-1,1])
            img = tensor2img(img, min_max=[-1,1])
            restored_img = np.concatenate([img, output], axis=1)
            restored_img = output
        del output
        torch.cuda.empty_cache()

        path = os.path.splitext(os.path.join(save_root, img_name))[0]
        save_path = f'{path}.png'
        imwrite(restored_img, save_path)

    print(f'\nAll results are saved in {save_root}')

