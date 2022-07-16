import argparse
import glob
import time
from xml.sax import parse
import numpy as np
import os
import cv2
import torch
import torchvision.transforms as transforms
from skimage import io
from basicsr.utils import imwrite, tensor2img
from basicsr.utils.face_util import FaceRestorationHelper
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()

    parser.add_argument('--w', type=float, default=0.5, help='Balance the quality and fidelity')
    parser.add_argument('--upscale_factor', type=int, default=2)
    parser.add_argument('--test_path', type=str, default='./inputs/cropped_faces')
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces')
    parser.add_argument('--upsample_num_times', type=int, default=1, help='Upsample the image before face detection')
    parser.add_argument('--save_inverse_affine', action='store_true')
    parser.add_argument('--only_keep_largest', action='store_true')
    parser.add_argument('--draw_box', action='store_true')

    # The following are the paths for dlib models
    parser.add_argument(
        '--detection_path', type=str,
        default='weights/dlib/mmod_human_face_detector-4cb19393.dat'
    )
    parser.add_argument(
        '--landmark5_path', type=str,
        default='weights/dlib/shape_predictor_5_face_landmarks-c4b1e980.dat'
    )
    parser.add_argument(
        '--landmark68_path', type=str,
        default='weights/dlib/shape_predictor_68_face_landmarks-fbdc2cb8.dat'
    )

    args = parser.parse_args()
    if args.test_path.endswith('/'):  # solve when path ends with /
        args.test_path = args.test_path[:-1]

    w = args.w
    result_root = f'results/{os.path.basename(args.test_path)}_{w}'

    # set up the Network
    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                            connect_list=['32', '64', '128', '256']).to(device)

    ckpt_path = 'weights/CodeFormer/codeformer.pth'
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()


    save_crop_root = os.path.join(result_root, 'cropped_faces')
    save_restore_root = os.path.join(result_root, 'restored_faces')
    save_final_root = os.path.join(result_root, 'final_results')
    save_input_root = os.path.join(result_root, 'inputs')

    face_helper = FaceRestorationHelper(args.upscale_factor, face_size=512)
    face_helper.init_dlib(args.detection_path, args.landmark5_path, args.landmark68_path)

    # scan all the jpg and png images
    for img_path in sorted(glob.glob(os.path.join(args.test_path, '*.[jp][pn]g'))):
        img_name = os.path.basename(img_path)
        print(f'Processing: {img_name}')
        if args.has_aligned: 
            # the input faces are already cropped and aligned
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.cropped_faces = [img]
            cropped_faces = face_helper.cropped_faces
        else:
            # detect faces
            num_det_faces = face_helper.detect_faces(
                img_path, upsample_num_times=args.upsample_num_times, only_keep_largest=args.only_keep_largest)
            # get 5 face landmarks for each face
            num_landmarks = face_helper.get_face_landmarks_5()
            print(f'\tDetect {num_det_faces} faces, {num_landmarks} landmarks.')
            # warp and crop each face
            save_crop_path = os.path.join(save_crop_root, img_name)
            face_helper.warp_crop_faces(save_crop_path, save_inverse_affine_path=None)
            cropped_faces = face_helper.cropped_faces

            # get 68 landmarks for each cropped face
            # num_landmarks = face_helper.get_face_landmarks_68()
            # print(f'\tDetect {num_landmarks} faces for 68 landmarks.')
            # assert len(cropped_faces) == len(face_helper.all_landmarks_68)

        # TODO
        # face_helper.free_dlib_gpu_memory()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(cropped_faces):
            # prepare data
            cropped_face = transforms.ToTensor()(cropped_face)
            cropped_face = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(cropped_face)
            cropped_face = cropped_face.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = net(cropped_face, w=w, adain=True)[0]
                    restored_face = tensor2img(output, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face, min_max=(-1, 1))

            path = os.path.splitext(os.path.join(save_restore_root, img_name))[0]
            if not args.has_aligned:
                save_path = f'{path}_{idx:02d}.png'
                face_helper.add_restored_face(restored_face)
            else:
                save_path = f'{path}.png'
            imwrite(restored_face, save_path)

        if not args.has_aligned:
            # paste each restored face to the input image
            face_helper.paste_faces_to_input_image(os.path.join(save_final_root, img_name), draw_box=args.draw_box)

        # clean all the intermediate results to process the next image
        face_helper.clean_all()

    print(f'\nAll results are saved in {result_root}')
