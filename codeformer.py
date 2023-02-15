import os
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from .basicsr.utils.download_util import load_file_from_url
from .basicsr.utils.misc import gpu_is_available, get_device
from .facelib.utils.face_restoration_helper import FaceRestoreHelper
from .facelib.utils.misc import is_gray
import numpy as np

from .basicsr.utils.registry import ARCH_REGISTRY
from PIL import Image
import sys
import pdb

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}

def set_realesrgan(bg_tile):
    from .basicsr.archs.rrdbnet_arch import RRDBNet
    from .basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available(): # set False in CPU/MPS mode
        no_half_gpu_list = ['1650', '1660'] # set False for GPUs that don't support f16
        if not True in [gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=model,
        tile=bg_tile,
        tile_pad=40,
        pre_pad=0,
        half=use_half
    )

    if not gpu_is_available():  # CPU
        import warnings
        warnings.warn('Running on CPU now! Make sure your PyTorch version matches your CUDA.'
                        'The unoptimized RealESRGAN is slow on CPU. '
                        'If you want to disable it, please remove `--bg_upsampler` and `--face_upsample` in command.',
                        category=RuntimeWarning)
    return upsampler

class CodeFormer():
    
    def __init__(self,
                 fidelity_weight = 0,
                 upsampling_weight = 2,
                 has_aligned = False,
                 only_center_face = False,
                 draw_box = False,
                 detection_model = 'retinaface_resnet50',
                 bg_upsampler = 'realesrgan',
                 face_upsample = False,
                 bg_tile = 400,
                 suffix = None,
                 ):
        """
        fidelity_weight: 0 to 1, 0 has the best quality but least fidelity. Default: 0
        upsampling_weight: The final upsampling scale of the image. Default: 2
        has_aligned: Input are cropped and aligned faces. Default: False
        only_center_face: Only return the face. Default: False
        draw_box: Draw the bounding box for the detected faces. Default: False
        detection_model: Face detector: Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                    Default: retinaface_resnet50'
        bg_upsampler: Background upsampler. Optional: realesrgan
        face_upsample: Face upsampler after enhancement. Default: False
        bg_tile: Tile size for background sampler. Default: 400
        suffix: Suffix of the restored faces. Default: None
        """
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = get_device()

        self.w = fidelity_weight
        self.upscale = upsampling_weight
        self.has_aligned = has_aligned
        self.only_center_face = only_center_face
        self.draw_box = draw_box
        self.detection_model = detection_model
        self.bg_upsampler = bg_upsampler
        self.face_upsample = face_upsample
        self.bg_tile = bg_tile
        self.suffix = suffix
    
    def process_images(self, imgs):
        # ------------------ set up background upsampler ------------------
        if self.bg_upsampler == 'realesrgan':
            bg_upsampler = set_realesrgan(self.bg_tile)
        else: 
            bg_upsampler = None

        # ------------------ set up face upsampler ------------------
        if self.face_upsample:
            if bg_upsampler is not None:
                face_upsampler = bg_upsampler
            else:
                face_upsampler = set_realesrgan(self.bg_tile)
        else:
            face_upsampler = None

        # ------------------ set up CodeFormer restorer -------------------
        net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, 
                                                connect_list=['32', '64', '128', '256']).to(self.device)
        
        
        # ckpt_path = 'weights/CodeFormer/codeformer.pth'
        ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], 
                                        model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()

        # ------------------ set up FaceRestoreHelper -------------------
        # large det_model: 'YOLOv5l', 'retinaface_resnet50'
        # small det_model: 'YOLOv5n', 'retinaface_mobile0.25'
        if not self.has_aligned: 
            print(f'Face detection model: {self.detection_model}')
        if bg_upsampler is not None: 
            print(f'Background upsampling: True, Face upsampling: {self.face_upsample}')
        else:
            print(f'Background upsampling: False, Face upsampling: {self.face_upsample}')

        face_helper = FaceRestoreHelper(
            self.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model = self.detection_model,
            save_ext='png',
            use_parse=True,
            device=self.device)
        
        restored_imgs = []

        # -------------------- start to processing ---------------------
        for i, img in enumerate(imgs):
            # clean all the intermediate results to process the next image
            face_helper.clean_all()
            
            if self.has_aligned: 
                # the input faces are already cropped and aligned
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
                face_helper.is_gray = is_gray(img, threshold=10)
                if face_helper.is_gray:
                    print('Grayscale input: True')
                face_helper.cropped_faces = [img]
            else:
                face_helper.read_image(img)
                # get face landmarks for each face
                num_det_faces = face_helper.get_face_landmarks_5(
                    only_center_face=self.only_center_face, resize=640, eye_dist_threshold=5)
                print(f'\tdetect {num_det_faces} faces')
                # align and warp each face
                face_helper.align_warp_face()

            # face restoration for each cropped face
            for idx, cropped_face in enumerate(face_helper.cropped_faces):
                # prepare data
                cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

                try:
                    with torch.no_grad():
                        output = net(cropped_face_t, w=self.w, adain=True)[0]
                        restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                    del output
                    torch.cuda.empty_cache()
                except Exception as error:
                    print(f'\tFailed inference for CodeFormer: {error}')
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                restored_face = restored_face.astype('uint8')
                face_helper.add_restored_face(restored_face, cropped_face)

            # paste_back
            if not self.has_aligned:
                # upsample the background
                if bg_upsampler is not None:
                    # Now only support RealESRGAN for upsampling background
                    bg_img = bg_upsampler.enhance(img, outscale=self.upscale)[0]
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)
                # paste each restored face to the input image
                if self.face_upsample and face_upsampler is not None: 
                    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.draw_box, face_upsampler=face_upsampler)
                else:
                    restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=self.draw_box)
            
            restored_imgs.append(restored_img)

        return restored_imgs    

if __name__ == '__main__':

    cf = CodeFormer(face_upsample = True)

    #Bring in a random image

    img = Image.open("/Users/bryanchia/Desktop/stanford/classes/cs/cs324/fm-fairness/cheerleader-0a.png")
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    img = cf.process_images([img])

    img = Image.fromarray(cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB))

    img.save("/Users/bryanchia/Desktop/stanford/classes/cs/cs324/fm-fairness/cheerleader-0a-fixed.png")