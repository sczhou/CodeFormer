import cv2
import math
import random
import numpy as np
import os.path as osp
from scipy.io import loadmat
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, 
                                        adjust_hue, adjust_saturation, normalize)
from basicsr.data import gaussian_kernels as gaussian_kernels
from basicsr.data.transforms import augment
from basicsr.data.data_util import paths_from_folder, brush_stroke_mask, random_ff_mask
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class FFHQBlindDataset(data.Dataset):

    def __init__(self, opt):
        super(FFHQBlindDataset, self).__init__()
        logger = get_root_logger()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        assert self.gt_size >= self.in_size, 'Wrong setting.'
        
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.component_path = opt.get('component_path', None)
        self.latent_gt_path = opt.get('latent_gt_path', None)

        if self.component_path is not None:
            self.crop_components = True
            self.components_dict = torch.load(self.component_path)
            self.eye_enlarge_ratio = opt.get('eye_enlarge_ratio', 1.4)
            self.nose_enlarge_ratio = opt.get('nose_enlarge_ratio', 1.1)
            self.mouth_enlarge_ratio = opt.get('mouth_enlarge_ratio', 1.3)
        else:
            self.crop_components = False

        if self.latent_gt_path is not None:
            self.load_latent_gt = True            
            self.latent_gt_dict = torch.load(self.latent_gt_path)
        else:
            self.load_latent_gt = False  

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = paths_from_folder(self.gt_folder)

        # inpainting mask
        self.gen_inpaint_mask = opt.get('gen_inpaint_mask', False)
        if self.gen_inpaint_mask:
            logger.info(f'generate mask ...')
            # self.mask_max_angle = opt.get('mask_max_angle', 10)
            # self.mask_max_len = opt.get('mask_max_len', 150)
            # self.mask_max_width = opt.get('mask_max_width', 50)
            # self.mask_draw_times = opt.get('mask_draw_times', 10)
            # # print
            # logger.info(f'mask_max_angle: {self.mask_max_angle}')
            # logger.info(f'mask_max_len: {self.mask_max_len}')
            # logger.info(f'mask_max_width: {self.mask_max_width}')
            # logger.info(f'mask_draw_times: {self.mask_draw_times}')

        # perform corrupt
        self.use_corrupt = opt.get('use_corrupt', True)
        self.use_motion_kernel = False
        # self.use_motion_kernel = opt.get('use_motion_kernel', True)

        if self.use_motion_kernel:
            self.motion_kernel_prob = opt.get('motion_kernel_prob', 0.001)
            motion_kernel_path = opt.get('motion_kernel_path', 'basicsr/data/motion-blur-kernels-32.pth')
            self.motion_kernels = torch.load(motion_kernel_path)

        if self.use_corrupt and not self.gen_inpaint_mask:
            # degradation configurations
            self.blur_kernel_size = opt['blur_kernel_size']
            self.blur_sigma = opt['blur_sigma']
            self.kernel_list = opt['kernel_list']
            self.kernel_prob = opt['kernel_prob']
            self.downsample_range = opt['downsample_range']
            self.noise_range = opt['noise_range']
            self.jpeg_range = opt['jpeg_range']
            # print
            logger.info(f'Blur: blur_kernel_size {self.blur_kernel_size}, sigma: [{", ".join(map(str, self.blur_sigma))}]')
            logger.info(f'Downsample: downsample_range [{", ".join(map(str, self.downsample_range))}]')
            logger.info(f'Noise: [{", ".join(map(str, self.noise_range))}]')
            logger.info(f'JPEG compression: [{", ".join(map(str, self.jpeg_range))}]')

        # color jitter
        self.color_jitter_prob = opt.get('color_jitter_prob', None)
        self.color_jitter_pt_prob = opt.get('color_jitter_pt_prob', None)
        self.color_jitter_shift = opt.get('color_jitter_shift', 20)
        if self.color_jitter_prob is not None:
            logger.info(f'Use random color jitter. Prob: {self.color_jitter_prob}, shift: {self.color_jitter_shift}')

        # to gray
        self.gray_prob = opt.get('gray_prob', 0.0)
        if self.gray_prob is not None:
            logger.info(f'Use random gray. Prob: {self.gray_prob}')
        self.color_jitter_shift /= 255.

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img


    def get_component_locations(self, name, status):
        components_bbox = self.components_dict[name]
        if status[0]:  # hflip
            # exchange right and left eye
            tmp = components_bbox['left_eye']
            components_bbox['left_eye'] = components_bbox['right_eye']
            components_bbox['right_eye'] = tmp
            # modify the width coordinate
            components_bbox['left_eye'][0] = self.gt_size - components_bbox['left_eye'][0]
            components_bbox['right_eye'][0] = self.gt_size - components_bbox['right_eye'][0]
            components_bbox['nose'][0] = self.gt_size - components_bbox['nose'][0]
            components_bbox['mouth'][0] = self.gt_size - components_bbox['mouth'][0]
        
        locations_gt = {}
        locations_in = {}
        for part in ['left_eye', 'right_eye', 'nose', 'mouth']:
            mean = components_bbox[part][0:2]
            half_len = components_bbox[part][2]
            if 'eye' in part:
                half_len *= self.eye_enlarge_ratio
            elif part == 'nose':
                half_len *= self.nose_enlarge_ratio
            elif part == 'mouth':
                half_len *= self.mouth_enlarge_ratio
            loc = np.hstack((mean - half_len + 1, mean + half_len))
            loc = torch.from_numpy(loc).float()
            locations_gt[part] = loc
            loc_in = loc/(self.gt_size//self.in_size)
            locations_in[part] = loc_in
        return locations_gt, locations_in


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # load gt image
        gt_path = self.paths[index]
        name = osp.basename(gt_path)[:-4]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)
        
        # random horizontal flip
        img_gt, status = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False, return_status=True)

        if self.load_latent_gt:
            if status[0]:
                latent_gt = self.latent_gt_dict['hflip'][name]
            else:
                latent_gt = self.latent_gt_dict['orig'][name]

        if self.crop_components:
            locations_gt, locations_in = self.get_component_locations(name, status)

        # generate in image
        img_in = img_gt
        if self.use_corrupt and not self.gen_inpaint_mask:
            # motion blur
            if self.use_motion_kernel and random.random() < self.motion_kernel_prob:
                m_i = random.randint(0,31)
                k = self.motion_kernels[f'{m_i:02d}']
                img_in = cv2.filter2D(img_in,-1,k)
            
            # gaussian blur
            kernel = gaussian_kernels.random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                self.blur_kernel_size,
                self.blur_sigma,
                self.blur_sigma, 
                [-math.pi, math.pi],
                noise_range=None)
            img_in = cv2.filter2D(img_in, -1, kernel)

            # downsample
            scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
            img_in = cv2.resize(img_in, (int(self.gt_size // scale), int(self.gt_size // scale)), interpolation=cv2.INTER_LINEAR)

            # noise
            if self.noise_range is not None:
                noise_sigma = np.random.uniform(self.noise_range[0] / 255., self.noise_range[1] / 255.)
                noise = np.float32(np.random.randn(*(img_in.shape))) * noise_sigma
                img_in = img_in + noise
                img_in = np.clip(img_in, 0, 1)

            # jpeg
            if self.jpeg_range is not None:
                jpeg_p = np.random.uniform(self.jpeg_range[0], self.jpeg_range[1])
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_p]
                _, encimg = cv2.imencode('.jpg', img_in * 255., encode_param)
                img_in = np.float32(cv2.imdecode(encimg, 1)) / 255.

            # resize to in_size
            img_in = cv2.resize(img_in, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)

        # if self.gen_inpaint_mask:
        #     inpaint_mask = random_ff_mask(shape=(self.gt_size,self.gt_size), 
        #         max_angle = self.mask_max_angle, max_len = self.mask_max_len, 
        #         max_width = self.mask_max_width, times = self.mask_draw_times)
        #     img_in = img_in * (1 - inpaint_mask.reshape(self.gt_size,self.gt_size,1)) + \
        #              1.0 * inpaint_mask.reshape(self.gt_size,self.gt_size,1)

        #     inpaint_mask = torch.from_numpy(inpaint_mask).view(1,self.gt_size,self.gt_size)

        if self.gen_inpaint_mask:
            img_in = (img_in*255).astype('uint8')
            img_in = brush_stroke_mask(Image.fromarray(img_in))
            img_in = np.array(img_in) / 255.

        # random color jitter (only for lq)
        if self.color_jitter_prob is not None and (np.random.uniform() < self.color_jitter_prob):
            img_in = self.color_jitter(img_in, self.color_jitter_shift)
        # random to gray (only for lq)
        if self.gray_prob and np.random.uniform() < self.gray_prob:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
            img_in = np.tile(img_in[:, :, None], [1, 1, 3])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_in, img_gt = img2tensor([img_in, img_gt], bgr2rgb=True, float32=True)

        # random color jitter (pytorch version) (only for lq)
        if self.color_jitter_pt_prob is not None and (np.random.uniform() < self.color_jitter_pt_prob):
            brightness = self.opt.get('brightness', (0.5, 1.5))
            contrast = self.opt.get('contrast', (0.5, 1.5))
            saturation = self.opt.get('saturation', (0, 1.5))
            hue = self.opt.get('hue', (-0.1, 0.1))
            img_in = self.color_jitter_pt(img_in, brightness, contrast, saturation, hue)

        # round and clip
        img_in = np.clip((img_in * 255.0).round(), 0, 255) / 255.

        # Set vgg range_norm=True if use the normalization here
        # normalize
        normalize(img_in, self.mean, self.std, inplace=True)
        normalize(img_gt, self.mean, self.std, inplace=True)

        return_dict = {'in': img_in, 'gt': img_gt, 'gt_path': gt_path}

        if self.crop_components:
            return_dict['locations_in'] = locations_in
            return_dict['locations_gt'] = locations_gt

        if self.load_latent_gt:
            return_dict['latent_gt'] = latent_gt

        # if self.gen_inpaint_mask:
        #     return_dict['inpaint_mask'] = inpaint_mask

        return return_dict


    def __len__(self):
        return len(self.paths)