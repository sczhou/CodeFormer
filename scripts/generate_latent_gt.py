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
    parser.add_argument('-o', '--save_root', type=str, default='./experiments/pretrained_models/vqgan')
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

    sum_latent = np.zeros((codebook_size)).astype('float64')
    size_latent = 16
    latent = {}
    latent['orig'] = {}
    latent['hflip'] = {}
    for i in ['orig', 'hflip']:
    # for i in ['hflip']:
        for img_path in sorted(glob.glob(os.path.join(test_path, '*.[jp][pn]g'))):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if i == 'hflip':
                cv2.flip(img, 1, img)
            img = img2tensor(img / 255., bgr2rgb=True, float32=True)
            normalize(img, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                # output = net(img)[0]
                x, feat_dict = vqgan.encoder(img, True)
                x, _, log = vqgan.quantize(x)
            # del output
            torch.cuda.empty_cache()

            min_encoding_indices = log['min_encoding_indices']
            min_encoding_indices = min_encoding_indices.view(size_latent,size_latent)
            latent[i][img_name[:-4]] = min_encoding_indices.cpu().numpy()
            print(img_name, latent[i][img_name[:-4]].shape)

    latent_save_path = os.path.join(save_root, f'latent_gt_code{codebook_size}.pth')
    torch.save(latent, latent_save_path)
    print(f'\nLatent GT code are saved in {save_root}')
