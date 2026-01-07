"""
This file is used for deploying hugging face demo:
https://huggingface.co/spaces/sczhou/CodeFormer
"""

import sys
import os

# root path
script_dir = os.path.dirname(os.path.abspath(__file__))  # web-demos/hugging_face
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

import cv2
import torch
import torch.nn.functional as F
import gradio as gr

from torchvision.transforms.functional import normalize

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import gpu_is_available, get_device
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY

from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray

# working directory & output directory
ui_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(ui_dir, 'output')
example_dir = os.path.join(ui_dir, 'examples')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(example_dir, exist_ok=True)

weights_dir = os.path.join(project_root, 'weights')

pretrain_model_url = {
    'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
    'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
    'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
}

# download weights (check in project weights directory)
print("Checking pre-trained models...")
weights_info = {
    'codeformer': (os.path.join(weights_dir, 'CodeFormer/codeformer.pth'), 'CodeFormer'),
    'detection': (os.path.join(weights_dir, 'facelib/detection_Resnet50_Final.pth'), 'detection'),
    'parsing': (os.path.join(weights_dir, 'facelib/parsing_parsenet.pth'), 'parsing'),
    'realesrgan': (os.path.join(weights_dir, 'realesrgan/RealESRGAN_x2plus.pth'), 'realesrgan'),
}

for model_name, (model_path, model_dir) in weights_info.items():
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        load_file_from_url(url=pretrain_model_url[model_name], 
                          model_dir=os.path.dirname(model_path), 
                          progress=True, file_name=None)
    else:
        print(f"✓ Found {model_name}")

# download example images (to examples directory)
print("Checking example images...")
example_files = [
    ('https://replicate.com/api/models/sczhou/codeformer/files/fa3fe3d1-76b0-4ca8-ac0d-0a925cb0ff54/06.png', '01.png'),
    ('https://replicate.com/api/models/sczhou/codeformer/files/a1daba8e-af14-4b00-86a4-69cec9619b53/04.jpg', '02.jpg'),
    ('https://replicate.com/api/models/sczhou/codeformer/files/542d64f9-1712-4de7-85f7-3863009a7c3d/03.jpg', '03.jpg'),
    ('https://replicate.com/api/models/sczhou/codeformer/files/a11098b0-a18a-4c02-a19a-9a7045d68426/010.jpg', '04.jpg'),
    ('https://replicate.com/api/models/sczhou/codeformer/files/7cf19c2c-e0cf-4712-9af8-cf5bdbb8d0ee/012.jpg', '05.jpg'),
]

for url, filename in example_files:
    example_path = os.path.join(example_dir, filename)
    if not os.path.exists(example_path):
        print(f"Downloading example {filename}...")
        try:
            torch.hub.download_url_to_file(url, example_path)
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
    else:
        print(f"✓ Found example {filename}")

def imread(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# set enhancer with RealESRGAN
def set_realesrgan():
    # half = True if torch.cuda.is_available() else False
    half = True if gpu_is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    realesrgan_path = os.path.join(weights_dir, "realesrgan/RealESRGAN_x2plus.pth")
    upsampler = RealESRGANer(
        scale=2,
        model_path=realesrgan_path,
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler

upsampler = set_realesrgan()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = get_device()
print("Loading CodeFormer model...")
codeformer_net = ARCH_REGISTRY.get("CodeFormer")(
    dim_embd=512,
    codebook_size=1024,
    n_head=8,
    n_layers=9,
    connect_list=["32", "64", "128", "256"],
).to(device)
ckpt_path = os.path.join(weights_dir, "CodeFormer/codeformer.pth")
checkpoint = torch.load(ckpt_path, map_location=device)["params_ema"]
codeformer_net.load_state_dict(checkpoint)
codeformer_net.eval()
print("CodeFormer model loaded successfully")

def inference(image, background_enhance, face_upsample, upscale, codeformer_fidelity):
    """Run a single prediction on the model"""
    try: # global try
        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"
        print('Inp:', image, background_enhance, face_upsample, upscale, codeformer_fidelity)

        img = cv2.imread(str(image), cv2.IMREAD_COLOR)
        print('\timage size:', img.shape)

        upscale = int(upscale) # convert type to int
        if upscale > 4: # avoid memory exceeded due to too large upscale
            upscale = 4 
        if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
            upscale = 2 
        if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            if face_helper.is_gray:
                print('\tgrayscale input: True')
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            print(f'\tdetect {num_det_faces} faces')
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )

        # save restored img
        save_path = os.path.join(output_dir, 'out.png')
        imwrite(restored_img, str(save_path))

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img, save_path
    except Exception as error:
        print('Global exception', error)
        return None, None


title = "CodeFormer: Robust Face Restoration and Enhancement Network"
description = r"""<center><img src='https://user-images.githubusercontent.com/14334509/189166076-94bb2cac-4f4e-40fb-a69f-66709e3d98f5.png' alt='CodeFormer logo'></center>
<b>Official Gradio demo</b> for <a href='https://github.com/sczhou/CodeFormer' target='_blank'><b>Towards Robust Blind Face Restoration with Codebook Lookup Transformer (NeurIPS 2022)</b></a>.<br>
🔥 CodeFormer is a robust face restoration algorithm for old photos or AI-generated faces.<br>
🤗 Try CodeFormer for improved stable-diffusion generation!<br>
"""
article = r"""
If CodeFormer is helpful, please help to ⭐ the <a href='https://github.com/sczhou/CodeFormer' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/sczhou/CodeFormer?style=social)](https://github.com/sczhou/CodeFormer)

---

📝 **Citation**

If our work is useful for your research, please consider citing:
```bibtex
@inproceedings{zhou2022codeformer,
    author = {Zhou, Shangchen and Chan, Kelvin C.K. and Li, Chongyi and Loy, Chen Change},
    title = {Towards Robust Blind Face Restoration with Codebook Lookup TransFormer},
    booktitle = {NeurIPS},
    year = {2022}
}
```

📋 **License**

This project is licensed under <a rel="license" href="https://github.com/sczhou/CodeFormer/blob/master/LICENSE">S-Lab License 1.0</a>. 
Redistribution and use for non-commercial purposes should follow this license.

📧 **Contact**

If you have any questions, please feel free to reach me out at <b>shangchenzhou@gmail.com</b>.

<div>
    🤗 Find Me:
    <a href="https://twitter.com/ShangchenZhou"><img style="margin-top:0.5em; margin-bottom:0.5em" src="https://img.shields.io/twitter/follow/ShangchenZhou?label=%40ShangchenZhou&style=social" alt="Twitter Follow"></a> 
    <a href="https://github.com/sczhou"><img style="margin-top:0.5em; margin-bottom:2em" src="https://img.shields.io/github/followers/sczhou?style=social" alt="Github Follow"></a>
</div>

<center><img src='https://visitor-badge-sczhou.glitch.me/badge?page_id=sczhou/CodeFormer' alt='visitors'></center>
"""

# load examples
examples = []
for i in range(1, 6):
    img_name = f'0{i}.png' if i <= 1 else f'0{i}.jpg' if i <= 4 else f'0{i}.jpg'
    example_path = os.path.join(example_dir, img_name)
    if os.path.exists(example_path):
        fidelity = 0.7 if i <= 3 else 0.1
        examples.append([example_path, True, True, 2, fidelity])

demo = gr.Interface(
    inference, [
        gr.Image(type="filepath", label="Input Image"),
        gr.Checkbox(value=True, label="Background Enhancement"),
        gr.Checkbox(value=True, label="Face Upsampling"),
        gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Upscale Factor"),
        gr.Slider(minimum=0, maximum=1, value=0.5, step=0.01, label='Fidelity (0=high quality, 1=preserve identity)')
    ], [
        gr.Image(type="numpy", label="Output Image"),
        gr.File(label="Download Result")
    ],
    title=title,
    description=description,
    article=article,
    examples=examples if examples else None
)

print("\n" + "="*60)
print("CodeFormer UI is starting...")
print("="*60)
print(f"Output directory: {output_dir}")
print(f"Example directory: {example_dir}")
print("="*60 + "\n")

# New Gradio API
try:
    demo.queue(max_threads=10)
except TypeError:
    # try old API
    try:
        demo.queue()
    except DeprecationWarning:
        # ignore
        pass

demo.launch()