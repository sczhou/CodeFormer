import cv2
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
# from basicsr.utils.download_util import download_file_from_google_drive

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def download_pretrained_models(file_ids, save_path_root):
    import gdown
    
    os.makedirs(save_path_root, exist_ok=True)

    for file_name, file_id in file_ids.items():
        file_url = 'https://drive.google.com/uc?id='+file_id
        save_path = osp.abspath(osp.join(save_path_root, file_name))
        if osp.exists(save_path):
            user_response = input(f'{file_name} already exist. Do you want to cover it? Y/N\n')
            if user_response.lower() == 'y':
                print(f'Covering {file_name} to {save_path}')
                gdown.download(file_url, save_path, quiet=False)
                # download_file_from_google_drive(file_id, save_path)
            elif user_response.lower() == 'n':
                print(f'Skipping {file_name}')
            else:
                raise ValueError('Wrong input. Only accepts Y/N.')
        else:
            print(f'Downloading {file_name} to {save_path}')
            gdown.download(file_url, save_path, quiet=False)
            # download_file_from_google_drive(file_id, save_path)


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py
    """
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(os.path.join(ROOT_DIR, model_dir), exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(ROOT_DIR, model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def color_diff(img):
    img = Image.fromarray(img)
    if len(img.getbands()) == 1:
        return True
    img1 = np.asarray(img.getchannel(channel=0), dtype=np.int16)
    img2 = np.asarray(img.getchannel(channel=1), dtype=np.int16)
    img3 = np.asarray(img.getchannel(channel=2), dtype=np.int16)
    diff1 = (img1 - img2).var()
    diff2 = (img2 - img3).var()
    diff3 = (img3 - img1).var()
    return (diff1 + diff2 + diff3) / 3.0

def is_gray(img, threshold=10):
    if color_diff(img) <= threshold:
        return True
    else:
        return False

def rgb2gray(img, out_channel=3):
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if out_channel == 3:
        gray = gray[:,:,np.newaxis].repeat(3, axis=2)
    return gray

def bgr2gray(img, out_channel=3):
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if out_channel == 3:
        gray = gray[:,:,np.newaxis].repeat(3, axis=2)
    return gray


def calc_mean_std(feat, eps=1e-5):
    """
    Args:
        feat (numpy): 3D [w h c]s
    """
    size = feat.shape
    assert len(size) == 3, 'The input feature should be 3D tensor.'
    c = size[2]
    feat_var = feat.reshape(-1, c).var(axis=0) + eps
    feat_std = np.sqrt(feat_var).reshape(1, 1, c)
    feat_mean = feat.reshape(-1, c).mean(axis=0).reshape(1, 1, c)
    return feat_mean, feat_std


def adain_npy(content_feat, style_feat):
    """Adaptive instance normalization for numpy.

    Args:
        content_feat (numpy): The input feature.
        style_feat (numpy): The reference feature.
    """
    size = content_feat.shape
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - np.broadcast_to(content_mean, size)) / np.broadcast_to(content_std, size)
    result_feat = normalized_feat * np.broadcast_to(style_std, size) + np.broadcast_to(style_mean, size)

    # Ensure values are within the range of source image bit depth
    bit_range = 256 if np.max(content_feat) < 256 else 65536 # determine 8 bit or 16 bit.
    a_min, a_max = np.min(result_feat, axis=(0,1)), np.max(result_feat, axis=(0,1)) # min/max of each color
    i_min, i_max = np.argmin(a_min), np.argmax(a_max) # find the color index of global min and max
    v_min, v_max = a_min[i_min], a_max[i_max]         # global min and max
    if v_max > bit_range or v_min < 0:                # pixel value is out of the range of bit depth
        # reduce the style_std to clamp values in range.
        mean_min, mean_max = style_mean[0][0][i_min], style_mean[0][0][i_max] # mean of color for min/max
        ratio = min(mean_min / (mean_min - v_min), (bit_range - 1e-12 - mean_max) / (v_max - mean_max))
        style_std = style_std * np.broadcast_to([ratio, ratio, ratio], style_std.shape)
        result_feat = normalized_feat * np.broadcast_to(style_std, size) + np.broadcast_to(style_mean, size)

    return result_feat
