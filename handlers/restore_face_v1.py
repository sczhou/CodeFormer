import base64

import cv2
import numpy as np
import torch
from flask import request, jsonify, Blueprint
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from basicsr.utils.registry import ARCH_REGISTRY
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from json_bodies.ImageRestoreFace import ImageRestoreFace
from utils.logs import log

pretrain_model_url = {
    'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
}


def set_realesrgan(bg_tile):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.utils.realesrgan_utils import RealESRGANer

    use_half = False
    if torch.cuda.is_available():
        no_half_gpu_list = ['1650', '1660']
        if not any(gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list):
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

    if not gpu_is_available():
        import warnings
        warnings.warn(
            'Running on CPU now! Make sure your PyTorch version matches your CUDA. The unoptimized RealESRGAN is slow '
            'on CPU.',
            category=RuntimeWarning)
    return upsampler


restore_face_route_v1 = Blueprint('restore_face_v1', __name__, url_prefix='/v1/restore_face')


@restore_face_route_v1.route("/image", methods=["POST"])
def restore_face():
    r = ImageRestoreFace(**request.get_json())
    device = get_device()

    bg_upsampler = set_realesrgan(r.bg_tile) if r.bg_upsampler == 'realesrgan' else None
    face_upsampler = bg_upsampler if r.face_upsampler else None

    net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                          connect_list=['32', '64', '128', '256']).to(device)
    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'], model_dir='models/CodeFormer', progress=True)
    checkpoint = torch.load(ckpt_path)['params_ema']
    net.load_state_dict(checkpoint)
    net.eval()

    face_helper = FaceRestoreHelper(
        r.upscale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model=r.detection_model,
        save_ext='png',
        use_parse=True,
        device=device
    )

    restored_images = []

    for img_b64 in r.input_images:
        face_helper.clean_all()
        img_data = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if r.has_aligned:
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=10)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            num_det_faces = face_helper.get_face_landmarks_5(only_center_face=r.only_center_face, resize=640,
                                                             eye_dist_threshold=5)
            log.info(f"Detected {num_det_faces} faces")
            face_helper.align_warp_face()

        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            with torch.no_grad():
                output = net(cropped_face_t, w=r.fidelity_weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            restored_face = restored_face.astype('uint8')
            face_helper.add_restored_face(restored_face, cropped_face)

        if not r.has_aligned:
            bg_img = bg_upsampler.enhance(img, outscale=r.upscale)[0] if bg_upsampler else None
            face_helper.get_inverse_affine(None)
            restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img, draw_box=r.draw_box,
                                                                  face_upsampler=face_upsampler) if face_upsampler else face_helper.paste_faces_to_input_image(
                upsample_img=bg_img, draw_box=r.draw_box)
            _, buffer = cv2.imencode('.png', restored_img)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            restored_images.append(img_b64)

    return jsonify(restored_images)
