import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .sr_model import SRModel


@MODEL_REGISTRY.register()
class CodeFormerModel(SRModel):
    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.input = data['in'].to(self.device)
        self.b = self.gt.shape[0]

        if 'latent_gt' in data:
            self.idx_gt = data['latent_gt'].to(self.device)
            self.idx_gt = self.idx_gt.view(self.b, -1)
        else:
            self.idx_gt = None

    def init_training_settings(self):
        logger = get_root_logger()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        if self.opt.get('network_vqgan', None) is not None and self.opt['datasets'].get('latent_gt_path') is None:
            self.hq_vqgan_fix = build_network(self.opt['network_vqgan']).to(self.device)
            self.hq_vqgan_fix.eval()
            self.generate_idx_gt = True
            for param in self.hq_vqgan_fix.parameters():
                param.requires_grad = False
        else:
            self.generate_idx_gt = False

        self.hq_feat_loss = train_opt.get('use_hq_feat_loss', True)
        self.feat_loss_weight = train_opt.get('feat_loss_weight', 1.0)
        self.cross_entropy_loss = train_opt.get('cross_entropy_loss', True)
        self.entropy_loss_weight = train_opt.get('entropy_loss_weight', 0.5)
        self.fidelity_weight = train_opt.get('fidelity_weight', 1.0)
        self.scale_adaptive_gan_weight = train_opt.get('scale_adaptive_gan_weight', 0.8)


        self.net_g.train()
        # define network net_d
        if self.fidelity_weight > 0:
            self.net_d = build_network(self.opt['network_d'])
            self.net_d = self.model_to_device(self.net_d)
            self.print_network(self.net_d)

            # load pretrained models
            load_path = self.opt['path'].get('pretrain_network_d', None)
            if load_path is not None:
                self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True))

            self.net_d.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)


        self.fix_generator = train_opt.get('fix_generator', True)
        logger.info(f'fix_generator: {self.fix_generator}')

        self.net_g_start_iter = train_opt.get('net_g_start_iter', 0)
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_start_iter = train_opt.get('net_d_start_iter', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def calculate_adaptive_weight(self, recon_loss, g_loss, last_layer, disc_weight_max):
        recon_grads = torch.autograd.grad(recon_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, disc_weight_max).detach()
        return d_weight

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_params_g = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params_g.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params_g, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        if self.fidelity_weight > 0:
            optim_type = train_opt['optim_d'].pop('type')
            self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
            self.optimizers.append(self.optimizer_d)

    def gray_resize_for_identity(self, out, size=128):
        out_gray = (0.2989 * out[:, 0, :, :] + 0.5870 * out[:, 1, :, :] + 0.1140 * out[:, 2, :, :])
        out_gray = out_gray.unsqueeze(1)
        out_gray = F.interpolate(out_gray, (size, size), mode='bilinear', align_corners=False)
        return out_gray

    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        if self.generate_idx_gt:
            x = self.hq_vqgan_fix.encoder(self.gt)
            output, _, quant_stats = self.hq_vqgan_fix.quantize(x)
            min_encoding_indices = quant_stats['min_encoding_indices']
            self.idx_gt = min_encoding_indices.view(self.b, -1)

        if self.fidelity_weight > 0:
            self.output, logits, lq_feat = self.net_g(self.input, w=self.fidelity_weight, detach_16=True)
        else:
            logits, lq_feat = self.net_g(self.input, w=0, code_only=True)

        if self.hq_feat_loss:
            # quant_feats
            quant_feat_gt = self.net_g.module.quantize.get_codebook_feat(self.idx_gt, shape=[self.b,16,16,256])

        l_g_total = 0
        loss_dict = OrderedDict()
        if current_iter % self.net_d_iters == 0 and current_iter > self.net_g_start_iter:
            # hq_feat_loss
            if self.hq_feat_loss: # codebook loss 
                l_feat_encoder = torch.mean((quant_feat_gt.detach()-lq_feat)**2) * self.feat_loss_weight
                l_g_total += l_feat_encoder
                loss_dict['l_feat_encoder'] = l_feat_encoder

            # cross_entropy_loss
            if self.cross_entropy_loss:
                # b(hw)n -> bn(hw)
                cross_entropy_loss = F.cross_entropy(logits.permute(0, 2, 1), self.idx_gt) * self.entropy_loss_weight
                l_g_total += cross_entropy_loss
                loss_dict['cross_entropy_loss'] = cross_entropy_loss

            if self.fidelity_weight > 0: # when fidelity_weight == 0 don't need image-level loss
                # pixel loss
                if self.cri_pix:
                    l_g_pix = self.cri_pix(self.output, self.gt)
                    l_g_total += l_g_pix
                    loss_dict['l_g_pix'] = l_g_pix

                # perceptual loss
                if self.cri_perceptual:
                    l_g_percep = self.cri_perceptual(self.output, self.gt)
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep

                # gan loss
                if  current_iter > self.net_d_start_iter:
                    fake_g_pred = self.net_d(self.output)
                    l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)
                    recon_loss = l_g_pix + l_g_percep
                    if not self.fix_generator:
                        last_layer = self.net_g.module.generator.blocks[-1].weight
                        d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)
                    else:
                        largest_fuse_size = self.opt['network_g']['connect_list'][-1]
                        last_layer = self.net_g.module.fuse_convs_dict[largest_fuse_size].shift[-1].weight
                        d_weight = self.calculate_adaptive_weight(recon_loss, l_g_gan, last_layer, disc_weight_max=1.0)
                    
                    d_weight *= self.scale_adaptive_gan_weight # 0.8
                    loss_dict['d_weight'] = d_weight
                    l_g_total += d_weight * l_g_gan
                    loss_dict['l_g_gan'] = d_weight * l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # optimize net_d
        if  current_iter > self.net_d_start_iter and self.fidelity_weight > 0:
            for p in self.net_d.parameters():
                p.requires_grad = True

            self.optimizer_d.zero_grad()
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
            loss_dict['l_d_real'] = l_d_real
            loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(self.output.detach())
            l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
            loss_dict['l_d_fake'] = l_d_fake
            loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()

            self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)


    def test(self):
        with torch.no_grad():
            if hasattr(self, 'net_g_ema'):
                self.net_g_ema.eval()
                self.output, _, _ = self.net_g_ema(self.input, w=self.fidelity_weight)
            else:
                logger = get_root_logger()
                logger.warning('Do not have self.net_g_ema, use self.net_g.')
                self.net_g.eval()
                self.output, _, _ = self.net_g(self.input, w=self.fidelity_weight)
                self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=sr_img, img2=gt_img)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)


    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)


    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['gt'] = self.gt.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        return out_dict


    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        if self.fidelity_weight > 0:
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)
