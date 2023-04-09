import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_colorization.pth'

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = get_device()
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', type=str, default='./inputs/gray_faces', 
                    help='Input image or folder. Default: inputs/gray_faces')
    parser.add_argument('-o', '--output_path', type=str, default=None, 
                    help='Output folder. Default: results/<input_name>')
    parser.add_argument('--suffix', type=str, default=None, 
                    help='Suffix of the restored faces. Default: None')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    print('[NOTE] The input face images should be aligned and cropped to a resolution of 512x512.')
    if args.input_path.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        input_img_list = [args.input_path]
        result_root = f'results/test_colorization_img'
    else: # input img folder
        if args.input_path.endswith('/'):  # solve when path ends with /
            args.input_path = args.input_path[:-1]
        # scan all the jpg and png images
        input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
        result_root = f'results/{os.path.basename(args.input_path)}'

    if not args.output_path is None: # set output path
        result_root = args.output_path

    test_img_num = len(input_img_list)

    # ------------------ set up CodeFormer restorer -------------------
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128']).to(device)
    
    # ckpt_path = 'weights/CodeFormer/codeformer.pth'
    ckpt_path = load_file_from_url(url=pretrain_model_url, 
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start to processing ---------------------
    for i, img_path in enumerate(input_img_list):
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        print(f'[{i+1}/{test_img_num}] Processing: {img_name}')
        input_face = cv2.imread(img_path)
        assert input_face.shape[:2] == (512, 512), 'Input resolution must be 512x512 for colorization.'
        # input_face = cv2.resize(input_face, (512, 512), interpolation=cv2.INTER_LINEAR)
        input_face = img2tensor(input_face / 255., bgr2rgb=True, float32=True)
        normalize(input_face, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        input_face = input_face.unsqueeze(0).to(device)
        try:
            with torch.no_grad():
                # w is fixed to 0 since we didn't train the Stage III for colorization
                output_face = net(input_face, w=0, adain=True)[0] 
                save_face = tensor2img(output_face, rgb2bgr=True, min_max=(-1, 1))
            del output_face
            torch.cuda.empty_cache()
        except Exception as error:
            print(f'\tFailed inference for CodeFormer: {error}')
            save_face = tensor2img(input_face, rgb2bgr=True, min_max=(-1, 1))

        save_face = save_face.astype('uint8')

        # save face
        if args.suffix is not None:
            basename = f'{basename}_{args.suffix}'
        save_restore_path = os.path.join(result_root, f'{basename}.png')
        imwrite(save_face, save_restore_path)

    print(f'\nAll results are saved in {result_root}')

