import torch
import torch.nn as nn
import torch.sparse as sparse
import torch.nn.functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)

import models.resnet38d

from tool import pyutils
from models.functions import ReverseLayerF
from models.functions import PAMR

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class Net(models.resnet38d.Net):
    def __init__(self, threshold=0.6):
        super(Net, self).__init__()

        self.threshold = threshold

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, 21, 1, bias=False)

        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192+3, 192, 1, bias=False)
        
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)
        
        self.discriminator = nn.Sequential()
        self.discriminator.add_module('d_conv1', nn.Conv2d(4096, 1024, 1))
        self.discriminator.add_module('d_relu1', nn.ReLU(inplace=True))
        self.discriminator.add_module('d_drop1', nn.Dropout())
        self.discriminator.add_module('d_conv2', nn.Conv2d(1024, 512, 1))
        self.discriminator.add_module('d_relu2', nn.ReLU(inplace=True))
        self.discriminator.add_module('d_drop2', nn.Dropout())
        torch.nn.init.xavier_uniform_(self.discriminator.d_conv1.weight)
        torch.nn.init.xavier_uniform_(self.discriminator.d_conv2.weight)

        self.from_scratch_layers = [self.f8_3, self.f8_4, self.f9, self.fc8, \
                                    self.discriminator.d_conv1, self.discriminator.d_conv2]

        self.head_list = []
        for c in range(20):
            head = nn.Conv2d(512, 2, 1)
            torch.nn.init.xavier_uniform_(head.weight)
            self.from_scratch_layers.append(head)
            self.head_list.append(head)
        
        self.head_list = nn.ModuleList(self.head_list)


        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]


        PAMR_ITER = 10
        PAMR_KERNEL = [1, 2, 4, 8, 12, 24]
        self.PAMR = PAMR(PAMR_ITER, PAMR_KERNEL)


    def forward(self, x, alpha=None, label=None):
        
        x_raw = self.denorm(x.clone())

        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        feature = d['conv6']
        cam = self.fc8(self.dropout7(feature))
      
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0

        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
        x_s = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        n,c,h,w = f.size()
        cam_rv = self.PCM(cam_d_norm, f)
        cam_rv_down = cam_rv.detach()
        cam_down = cam
        cam_rv = F.interpolate(cam_rv, (H,W), mode='bilinear', align_corners=True)
        cam = F.interpolate(cam, (H,W), mode='bilinear', align_corners=True)

        if alpha is None:
            return cam, cam_rv

        # reversied_feature
        reversed_feature = ReverseLayerF.apply(feature, alpha)
        domain_fea = self.discriminator(reversed_feature)

        domain_logits = {}

        for c in range(20):
            domain_logits[c] = self.head_list[c](domain_fea)  # (n, 2, h, w)

        logits = F.relu(cam_rv_down)
        s_mask, s_weights = self.source_mask(logits, label, thres=self.threshold)  # (n, c, h, w)

        masked_x = (1 - F.interpolate((s_mask.sum(1)>1e-5).float().unsqueeze(1), (H, W))) * x
        t_mask, t_weights, t_pseudo_gt = self.masked_forward(masked_x, label, thres=self.threshold) # (n, h, w)
        
        domain_labels = {}
        s_domain_weights = {}
        t_domain_weights = {}
        for c in range(20):
            domain_label_c = torch.ones(n, h, w) * 255
            s_mask_c = s_mask[:, c, :, :]
            t_mask_c = t_mask[:, c, :, :]
            s_weights_c = s_weights[:, c, :, :]  # weights
            t_weights_c = t_weights[:, c, :, :]  # (n, h, w)
            
            domain_label_c[t_mask_c == 1] = 1
            domain_label_c[s_mask_c == 1] = 0

            domain_labels[c] = domain_label_c.long().cuda()
            s_domain_weights[c] = s_weights_c
            t_domain_weights[c] = t_weights_c
        
        cam_logits = cam_down.detach()
        cam_logits[:, 0, :, :] = torch.ones_like(cam_logits[:, 0])
        cam_logits = self.run_pamr(x_raw, torch.softmax(cam_logits, dim=1))
        cam_logits *= label
        pseudo_gt = self.pseudo_gtmask(cam_logits, cutoff_top=0.6, cutoff_top_bg=0.7).detach()
        bs, c, h, w = pseudo_gt.shape
        fg_mask = torch.cat([torch.zeros(bs, 1, h, w).cuda(), s_mask.float()], dim=1)
        pseudo_gt = (pseudo_gt * fg_mask).type_as(pseudo_gt)
        t_fg_mask = torch.cat([torch.zeros(bs, 1, h, w).cuda(), t_mask.float()], dim=1)
        t_pseudo_gt = (t_pseudo_gt * t_fg_mask).type_as(t_pseudo_gt)


        return cam, cam_rv, domain_logits, domain_labels, s_domain_weights, t_domain_weights, cam_down, pseudo_gt, t_pseudo_gt




    @torch.no_grad()
    def masked_forward(self, x, label, thres=0.6):
        
        x_raw = self.denorm(x.clone())

        N, C, H, W = x.size()
        d = super().forward_as_dict(x)
        feature = d['conv6']
        cam = self.fc8(self.dropout7(feature))
        n,c,h,w = cam.size()
        with torch.no_grad():
            cam_d = F.relu(cam.detach())
            cam_d_max = torch.max(cam_d.view(n,c,-1), dim=-1)[0].view(n,c,1,1)+1e-5
            cam_d_norm = F.relu(cam_d-1e-5)/cam_d_max
            cam_d_norm[:,0,:,:] = 1-torch.max(cam_d_norm[:,1:,:,:], dim=1)[0]
            cam_max = torch.max(cam_d_norm[:,1:,:,:], dim=1, keepdim=True)[0]
            cam_d_norm[:,1:,:,:][cam_d_norm[:,1:,:,:] < cam_max] = 0

        f8_3 = F.relu(self.f8_3(d['conv4'].detach()), inplace=True)
        f8_4 = F.relu(self.f8_4(d['conv5'].detach()), inplace=True)
        x_s = F.interpolate(x,(h,w),mode='bilinear',align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)
        n,c,h,w = f.size()
        cam_rv = self.PCM(cam_d_norm, f)
        cam_rv_down = cam_rv.detach()        
        logits = F.relu(cam_rv_down)        
        s_mask, s_weights = self.source_mask(logits, label, thres=thres)

        cam_logits = cam.detach()
        cam_logits[:, 0, :, :] = torch.ones_like(cam_logits[:, 0])
        cam_logits = self.run_pamr(x_raw, torch.softmax(cam_logits, dim=1))
        cam_logits *= label
        pseudo_gt = self.pseudo_gtmask(cam_logits, cutoff_top=0.6, cutoff_top_bg=0.7).detach() 
        bs, c, h, w = pseudo_gt.shape
        fg_mask = torch.cat([torch.zeros(bs, 1, h, w).cuda(), s_mask.float()], dim=1) 
        pseudo_gt = (pseudo_gt * fg_mask).type_as(pseudo_gt)

        return s_mask, s_weights, pseudo_gt
    
    def source_mask(self, mask, label, thres=0.6, ratio=0.6, eps=1e-5):

        mask = mask[:, 1:, :, :]
        bs,c,h,w = mask.size()

        max1 = torch.max(mask.view(bs, c, -1), dim=-1)[0].view(bs, c, 1, 1)
        min1 = torch.min(mask.view(bs, c, -1), dim=-1)[0].view(bs, c, 1, 1)
        mask[mask < min1 + 1e-5] = 0.
        mask = (mask - min1 - 1e-5) / (max1 - min1 + 1e-5)
        mask = mask * label[:, 1:, :, :]
        source_mask = mask > thres

        return source_mask, mask


    def pseudo_gtmask(self, mask, cutoff_top=0.8, cutoff_top_bg=0.9, cutoff_low=0.2, eps=1e-8):

        bs,c,h,w = mask.size()
        mask = mask.view(bs,c,-1)

        mask_max, _ = mask.max(-1, keepdim=True)
        mask_max[:, :1] *= cutoff_top_bg
        mask_max[:, 1:] *= cutoff_top

        lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
        mask_max = mask_max.max(lowest)

        pseudo_gt = (mask > mask_max).type_as(mask)

        ambiguous = (pseudo_gt.sum(1, keepdim=True) > 1).type_as(mask)
        pseudo_gt = (1 - ambiguous) * pseudo_gt
 
        return pseudo_gt.view(bs,c,h,w)

    def balanced_mask_loss_ce(self, mask, pseudo_gt, ignore_index=255):

        mask = F.interpolate(mask, size=pseudo_gt.size()[-2:], mode="bilinear", align_corners=True)
        
        # indices of the max classes
        mask_gt = torch.argmax(pseudo_gt, 1)

        # for each pixel there should be at least one 1
        # otherwise, ignore
        ignore_mask = pseudo_gt.sum(1) < 1.
        mask_gt[ignore_mask] = ignore_index

        # class weight balances the loss w.r.t. number of pixels
        # because we are equally interested in all classes
        bs,c,h,w = pseudo_gt.size()
        num_pixels_per_class = pseudo_gt.view(bs,c,-1).sum(-1)
        num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)
        class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)
        class_weight = (pseudo_gt * class_weight[:,:,None,None]).sum(1).view(bs, -1)
        loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none")

        return loss.mean()

    def PCM(self, cam, f):
        n,c,h,w = f.size()
        cam = F.interpolate(cam, (h,w), mode='bilinear', align_corners=True).view(n,-1,h*w)
        f = self.f9(f)
        f = f.view(n,-1,h*w)
        f = f/(torch.norm(f,dim=1,keepdim=True)+1e-5)

        aff = F.relu(torch.matmul(f.transpose(1,2), f),inplace=True)
        aff = aff/(torch.sum(aff,dim=1,keepdim=True)+1e-5)
        cam_rv = torch.matmul(cam, aff).view(n,-1,h,w)
        
        return cam_rv

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)) or isinstance(m, nn.Linear):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups

    def run_pamr(self, im, mask):
        im = F.interpolate(im, mask.size()[-2:], mode="bilinear", align_corners=True)  # downsample the image
        masks_dec = self.PAMR(im, mask)
        return masks_dec

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, MEAN, STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), MEAN, STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image



