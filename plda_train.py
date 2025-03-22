import numpy as np
import torch
import random
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import voc12.data
from tool import pyutils, imutils, torchutils, visualization
import argparse
import importlib
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import traceback


def adaptive_min_pooling_loss(x):
    # This loss does not affect the highest performance, but change the optimial background score (alpha)
    n,c,h,w = x.size()
    k = h*w//4
    x = torch.max(x, dim=1)[0]
    y = torch.topk(x.view(n,-1), k=k, dim=-1, largest=False)[0]
    y = F.relu(y, inplace=False)
    loss = torch.sum(y)/(k*n)
    return loss

def max_onehot(x):
    n,c,h,w = x.size()
    x_max = torch.max(x[:,1:,:,:], dim=1, keepdim=True)[0]
    x[:,1:,:,:][x[:,1:,:,:] != x_max] = 0
    return x
    
def balanced_mask_loss_ce(mask, pseudo_gt, ignore_index=255):
    """Class-balanced CE loss
    - cancel loss if only one class in pseudo_gt
    - weight loss equally between classes
    """

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

    # BCE loss
    loss = F.cross_entropy(mask, mask_gt, ignore_index=ignore_index, reduction="none").view(bs, -1)

    # reweight by class weights
    loss = (class_weight * loss).mean()
        
    return loss


def balanced_weighted_multi_adv_domain_loss(logits_, label_, s_weights_, t_weights_, ignore_index=255):
    loss_total = torch.tensor(0).cuda()
    cnt = 0
    for c in range(20):
        logits = logits_[c]
        label = label_[c]

        if torch.sum(label < 255).item() < 1e-5:
            continue

        bs, h, w = label.shape 
        s_weights = s_weights_[c]
        t_weights = t_weights_[c]
        logits = F.interpolate(logits, size=label.size()[-2:], mode='bilinear', align_corners=True)
        loss = F.cross_entropy(logits, label, ignore_index=ignore_index, reduction="none").view(bs, -1)
        
        # logits: (n, c, h, w)
        prob = F.softmax(logits.detach(), dim=1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
        entropy_weights = 1 + torch.exp(-1 * entropy)  # (bs, h, w)
        entropy_weights = entropy_weights.view(bs, -1)
        loss = entropy_weights * loss

        # step 2: 平衡source/target
        source_num = torch.sum(label.view(bs, -1) == 0, dim=-1, keepdim=True)  # (bs, 1)
        target_num = torch.sum(label.view(bs, -1) == 1, dim=-1, keepdim=True)  # (bs, 1)
        num_pixels_per_class = torch.cat([source_num, target_num], dim=-1) # (bs, 2)
        num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)  # (bs, 1)
        class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)  # (bs, 2)
        one_hot = torch.zeros(bs, 2, h, w).cuda()
        one_hot[:, 0, :, :][label == 0] = 1
        one_hot[:, 1, :, :][label == 1] = 1
        class_weight = (one_hot * class_weight[:,:,None,None]).sum(1).view(bs, -1)  # (bs, hw)
        loss = class_weight * loss
        loss_total = loss_total + loss.mean()
        cnt += 1

    return loss_total


def balanced_domain_loss(logits, label, ignore_index=255):
    logits = F.interpolate(logits, size=label.size()[-2:], mode='bilinear', align_corners=True)
    bs, h, w = label.shape
    source_num = torch.sum(label.view(bs, -1) == 0, dim=-1, keepdim=True)  # (bs, 1)
    target_num = torch.sum(label.view(bs, -1) == 1, dim=-1, keepdim=True)  # (bs, 1)
    num_pixels_per_class = torch.cat([source_num, target_num], dim=-1) # (bs, 2)
    num_pixels_total = num_pixels_per_class.sum(-1, keepdim=True)  # (bs, 1)
    class_weight = (num_pixels_total - num_pixels_per_class) / (1 + num_pixels_total)  # (bs, 2)
    one_hot = torch.zeros(bs, 2, h, w).cuda()
    one_hot[:, 0, :, :][label == 0] = 1
    one_hot[:, 1, :, :][label == 1] = 1
    class_weight = (one_hot * class_weight[:,:,None,None]).sum(1).view(bs, -1)  # (bs, hw)
    loss = F.cross_entropy(logits, label, ignore_index=ignore_index, reduction="none").view(bs, -1)  # (bs, hw)
    
    loss = (class_weight * loss).mean()
    return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_epoches", default=8, type=int)
    parser.add_argument("--network", default="", type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument("--train_list", default="voc12/train_aug.txt", type=str)
    parser.add_argument("--val_list", default="voc12/val.txt", type=str)
    parser.add_argument("--session_name", default="", type=str)
    parser.add_argument("--crop_size", default=448, type=int)
    parser.add_argument("--weights", required=True, type=str)
    parser.add_argument("--voc12_root", default='VOC2012', type=str)
    parser.add_argument("--tblog_dir", default='./tblog', type=str)
    parser.add_argument("--gamma", default=1.0, type=float)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--threshold", default=0.6, type=float)
    args = parser.parse_args()

    pyutils.Logger(args.session_name + '.log')

    print(vars(args))
    
    model = getattr(importlib.import_module(args.network), 'Net')(args.threshold)

    print(model)

    tblogger = SummaryWriter(args.tblog_dir)	

    train_dataset = voc12.data.VOC12ClsDataset(args.train_list, voc12_root=args.voc12_root,
                                               transform=transforms.Compose([
                        imutils.RandomResizeLong(448, 768),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                        np.asarray,
                        model.normalize,
                        imutils.RandomCrop(args.crop_size),
                        imutils.HWC_to_CHW,
                        torch.from_numpy
                    ]))
    def worker_init_fn(worker_id):
        np.random.seed(1 + worker_id)
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True,
                                   worker_init_fn=worker_init_fn)
    max_step = len(train_dataset) // args.batch_size * args.max_epoches

    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)

    if args.weights[-7:] == '.params':
        import models.resnet38d
        # assert 'resnet38' in args.network
        weights_dict = models.resnet38d.convert_mxnet_to_torch(args.weights)
    else:
        weights_dict = torch.load(args.weights)

    model.load_state_dict(weights_dict, strict=False)
    model = torch.nn.DataParallel(model).cuda()
    model.train()

    avg_meter = pyutils.AverageMeter('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_adv1', 'loss_adv2', 'loss_ce1', 'loss_ce2', 'loss_t_ce1', 'loss_t_ce2')

    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):

        for iter, pack in enumerate(train_data_loader):
            
            p = float(iter + ep * len(train_data_loader)) / args.max_epoches / len(train_data_loader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            scale_factor = 0.3
            img1 = pack[1]
            img2 = F.interpolate(img1,scale_factor=scale_factor,mode='bilinear',align_corners=True) 
            N,C,H,W = img1.size()
            label = pack[2]
            bg_score = torch.ones((N,1))
            label = torch.cat((bg_score, label), dim=1)
            label = label.cuda(non_blocking=True).unsqueeze(2).unsqueeze(3)
            cam1, cam_rv1, domain_logits1, domain_label1, s_weights1, t_weights1, cam_down1, pseudo_gt1, t_pseudo_gt1 = model(img1, alpha, label)
            label1 = F.adaptive_avg_pool2d(cam1, (1,1))
            loss_rvmin1 = adaptive_min_pooling_loss((cam_rv1*label)[:,1:,:,:])
            cam1 = F.interpolate(visualization.max_norm(cam1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label
            cam_rv1 = F.interpolate(visualization.max_norm(cam_rv1),scale_factor=scale_factor,mode='bilinear',align_corners=True)*label

            cam2, cam_rv2, domain_logits2, domain_label2, s_weights2, t_weights2, cam_down2, pseudo_gt2, t_pseudo_gt2 = model(img2, alpha, label)
            label2 = F.adaptive_avg_pool2d(cam2, (1,1))
            loss_rvmin2 = adaptive_min_pooling_loss((cam_rv2*label)[:,1:,:,:])
            cam2 = visualization.max_norm(cam2)*label
            cam_rv2 = visualization.max_norm(cam_rv2)*label

            loss_cls1 = F.multilabel_soft_margin_loss(label1[:,1:,:,:], label[:,1:,:,:])
            loss_cls2 = F.multilabel_soft_margin_loss(label2[:,1:,:,:], label[:,1:,:,:])

   

            ns,cs,hs,ws = cam2.size()
            loss_er = torch.mean(torch.abs(cam1[:,1:,:,:]-cam2[:,1:,:,:]))
            cam1[:,0,:,:] = 1-torch.max(cam1[:,1:,:,:],dim=1)[0]
            cam2[:,0,:,:] = 1-torch.max(cam2[:,1:,:,:],dim=1)[0]
            tensor_ecr1 = torch.abs(max_onehot(cam2.detach()) - cam_rv1)
            tensor_ecr2 = torch.abs(max_onehot(cam1.detach()) - cam_rv2)
            loss_ecr1 = torch.mean(torch.topk(tensor_ecr1.view(ns,-1), k=(int)(21*hs*ws*0.2), dim=-1)[0])
            loss_ecr2 = torch.mean(torch.topk(tensor_ecr2.view(ns,-1), k=(int)(21*hs*ws*0.2), dim=-1)[0])
            loss_ecr = loss_ecr1 + loss_ecr2

            loss_cls = (loss_cls1 + loss_cls2)/2 + (loss_rvmin1 + loss_rvmin2)/2 
            loss = loss_cls + loss_er + loss_ecr
            
            loss_adv1 = balanced_weighted_multi_adv_domain_loss(domain_logits1, domain_label1, s_weights1, t_weights1)
            loss_adv2 = balanced_weighted_multi_adv_domain_loss(domain_logits2, domain_label2, s_weights2, t_weights2)

            loss_ce1 = balanced_mask_loss_ce(cam_down1, pseudo_gt1) 
            loss_ce2 = balanced_mask_loss_ce(cam_down2, pseudo_gt2)

            loss_t_ce1 = balanced_mask_loss_ce(cam_down1, t_pseudo_gt1)
            loss_t_ce2 = balanced_mask_loss_ce(cam_down2, t_pseudo_gt2)


            loss_adv = (loss_adv1 + loss_adv2) / 2
            loss_ce = (loss_ce1 + loss_ce2) / 2
            loss_t_ce = (loss_t_ce1 + loss_t_ce2) / 2
            loss = loss + loss_adv + loss_ce + loss_t_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_meter.add({
                'loss': loss.item(), 
                'loss_cls': loss_cls.item(), 
                'loss_er': loss_er.item(), 
                'loss_ecr': loss_ecr.item(),
                'loss_adv1': loss_adv1.item(),
                'loss_adv2': loss_adv2.item(),
                'loss_ce1': loss_ce1.item(),
                'loss_ce2': loss_ce2.item(),
                'loss_t_ce1': loss_t_ce1.item(),
                'loss_t_ce2': loss_t_ce2.item()
                })

            if (optimizer.global_step - 1) % 50 == 0:

                timer.update_progress(optimizer.global_step / max_step)

                print('Iter:%5d/%5d' % (optimizer.global_step-1, max_step),
                      'loss:%.4f | loss_cls:%.4f loss_er:%.4f loss_ecr:%.4f loss_adv1:%.4f loss_adv2:%.4f loss_ce1:%.4f loss_ce2:%.4f loss_t_ce1:%.4f loss_t_ce2:%.4f' % avg_meter.get('loss', 'loss_cls', 'loss_er', 'loss_ecr', 'loss_adv1', 'loss_adv2', 'loss_ce1', 'loss_ce2', 'loss_t_ce1', 'loss_t_ce2'),
                      'imps:%.1f' % ((iter+1) * args.batch_size / timer.get_stage_elapsed()),
                      'Fin:%s' % (timer.str_est_finish()),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

                avg_meter.pop()

                # Visualization for training process
                img_8 = img1[0].numpy().transpose((1,2,0))
                img_8 = np.ascontiguousarray(img_8)
                mean = (0.485, 0.456, 0.406)
                std = (0.229, 0.224, 0.225)
                img_8[:,:,0] = (img_8[:,:,0]*std[0] + mean[0])*255
                img_8[:,:,1] = (img_8[:,:,1]*std[1] + mean[1])*255
                img_8[:,:,2] = (img_8[:,:,2]*std[2] + mean[2])*255
                img_8[img_8 > 255] = 255
                img_8[img_8 < 0] = 0
                img_8 = img_8.astype(np.uint8)

                input_img = img_8.transpose((2,0,1))
                h = H//4; w = W//4
                p1 = F.interpolate(cam1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                p2 = F.interpolate(cam2,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                p_rv1 = F.interpolate(cam_rv1,(h,w),mode='bilinear')[0].detach().cpu().numpy()
                p_rv2 = F.interpolate(cam_rv2,(h,w),mode='bilinear')[0].detach().cpu().numpy()

                image = cv2.resize(img_8, (w,h), interpolation=cv2.INTER_CUBIC).transpose((2,0,1))
                CLS1, CAM1, _, _ = visualization.generate_vis(p1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                CLS2, CAM2, _, _ = visualization.generate_vis(p2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                CLS_RV1, CAM_RV1, _, _ = visualization.generate_vis(p_rv1, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                CLS_RV2, CAM_RV2, _, _ = visualization.generate_vis(p_rv2, None, image, func_label2color=visualization.VOClabel2colormap, threshold=None, norm=False)
                #MASK = eq_mask[0].detach().cpu().numpy().astype(np.uint8)*255
                loss_dict = {'loss':loss.item(), 
                             'loss_cls':loss_cls.item(),
                             'loss_er':loss_er.item(),
                             'loss_ecr':loss_ecr.item(),
                             'loss_adv1':loss_adv1.item(),
                             'loss_adv2':loss_adv2.item()}
                itr = optimizer.global_step - 1
                tblogger.add_scalars('loss', loss_dict, itr)
                tblogger.add_scalar('lr', optimizer.param_groups[0]['lr'], itr)
                tblogger.add_image('Image', input_img, itr)
                #tblogger.add_image('Mask', MASK, itr)
                tblogger.add_image('CLS1', CLS1, itr)
                tblogger.add_image('CLS2', CLS2, itr)
                tblogger.add_image('CLS_RV1', CLS_RV1, itr)
                tblogger.add_image('CLS_RV2', CLS_RV2, itr)
                tblogger.add_images('CAM1', CAM1, itr)
                tblogger.add_images('CAM2', CAM2, itr)
                tblogger.add_images('CAM_RV1', CAM_RV1, itr)
                tblogger.add_images('CAM_RV2', CAM_RV2, itr)

        else:
            print('')
            timer.reset_stage()

    torch.save(model.module.state_dict(), args.session_name + '.pth')         