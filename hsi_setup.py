from email.mime import base, image
from locale import normalize
from math import fabs
from xml.sax import SAXException
import torch
import torch.optim as optim
import models

import os
import argparse

from os.path import join
from utility import *
from utility.ssim import SSIMLoss,SAMLoss
from thop import profile
from torchstat import stat 
import scipy.io as scio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import models as torchmodel
from torch import  einsum

import torchvision.utils as vutil

import time
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

class MultipleLoss(nn.Module):
    def __init__(self, losses, weight=None):
        super(MultipleLoss, self).__init__()
        self.losses = nn.ModuleList(losses)
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target):
        total_loss = 0
        for weight, loss in zip(self.weight, self.losses):
            total_loss += loss(predict, target) * weight
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)

class L1Consist(nn.Module):
    def __init__(self, losses, weight=None):
        super(L1Consist, self).__init__()
        self.loss1 = losses[0]
        self.loss_cons = losses[1]
        self.weight = weight or [1/len(self.losses)] * len(self.losses)
    
    def forward(self, predict, target,inputs):
        total_loss = 0
        total_loss += self.loss1(predict, target) * self.weight[0]
        total_loss += self.loss_cons( predict , target,inputs) * self.weight[1]
        return total_loss

    def extra_repr(self):
        return 'weight={}'.format(self.weight)

def train_options(parser):
    def _parse_str_args(args):
        str_args = args.split(',')
        parsed_args = []
        for str_arg in str_args:
            arg = int(str_arg)
            if arg >= 0:
                parsed_args.append(arg)
        return parsed_args    
    parser.add_argument('--prefix', '-p', type=str, default='denoise',
                        help='prefix')
    parser.add_argument('--arch', '-a', metavar='ARCH', required=True,
                        choices=                                                         model_names        ,
                        help='model architecture: ' +
                        ' | '.join(model_names))
    parser.add_argument('--batchSize', '-b', type=int,
                        default=16, help='training batch size. default=16')         
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate. default=1e-3.')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight decay. default=0')
    parser.add_argument('--loss', type=str, default='l2',
                        help='which loss to choose.', choices=['l1', 'l2', 'smooth_l1', 'ssim', 'l2_ssim','l2_sam','cons','cons_l2'])
    parser.add_argument('--testdir', type=str)
    parser.add_argument('--sigma', type=int)

    parser.add_argument('--init', type=str, default='kn',
                        help='which init scheme to choose.', choices=['kn', 'ku', 'xn', 'xu', 'edsr'])
    parser.add_argument('--no-cuda', action='store_true', help='disable cuda?')
    parser.add_argument('--no-log', action='store_true',
                        help='disable logger?')
    parser.add_argument('--threads', type=int, default=0,
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2018,
                        help='random seed to use. default=2018')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--no-ropt', '-nro', action='store_true',
                            help='not resume optimizer')          
    parser.add_argument('--chop', action='store_true',
                            help='forward chop')                                      
    parser.add_argument('--resumePath', '-rp', type=str,
                        default=None, help='checkpoint to use.')
    parser.add_argument('--test-dir', type=str,
                        default='/data/HSI_Data/test_noise_96_icvl/512_10', help='The path of test HSIs')
    parser.add_argument('--resultdir', type=str,
                        default='/data/HSI_Data/Hyperspectral_Projec/imgs/default', help='The path of test HSIs')
    parser.add_argument('--dataroot', '-d', type=str,
                        default='/root/autodl-tmp/ICVL64_31.db', help='data root')
    parser.add_argument('--clip', type=float, default=1e6)
    parser.add_argument('--gpu-ids', type=str, default='0', help='gpu ids')

    ####################
    parser.add_argument('--update_lr', type=float, default=0.5e-4, help='learning rate of inner loop')
    parser.add_argument('--meta_lr', type=float, default=0.5e-4, help='learning rate of outer loop')
    parser.add_argument('--n_way', type=int, default=1, help='the number of ways')
    parser.add_argument('--k_spt', type=int, default=2, help='the number of support set')
    parser.add_argument('--k_qry', type=int, default=5, help='the number of query set')
    parser.add_argument('--task_num', type=int, default=16, help='the number of tasks')
    parser.add_argument('--update_step', type=int, default=5, help='update step of inner loop in training')
    parser.add_argument('--update_step_test', type=int, default=10, help='update step of inner loop in testing')
    opt = parser.parse_args()
    opt.gpu_ids = _parse_str_args(opt.gpu_ids)

    return opt


def make_dataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    # dataset.length -= 1000
    # dataset.length = size or dataset.length

    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = ImageTransformDataset(dataset, train_transform, target_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)

    return train_loader


def make_metadataset(opt, train_transform, target_transform, common_transform, batch_size=None, repeat=1):
    dataset = LMDBDataset(opt.dataroot, repeat=repeat)
    # dataset.length -= 1000
    # dataset.length = size or dataset.length

    """Split patches dataset into training, validation parts"""
    dataset = TransformDataset(dataset, common_transform)

    train_dataset = MetaRandomDataset(dataset, opt.n_way, opt.k_spt, opt.k_qry, train_transform, target_transform)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size or opt.batchSize, shuffle=True,
                              num_workers=opt.threads, pin_memory=not opt.no_cuda, worker_init_fn=worker_init_fn)
    
    return train_loader

class Engine(object):
    def __init__(self, opt):
        self.prefix = opt.prefix
        self.opt = opt
        self.net = None
        self.optimizer = None
        self.criterion = None
        self.basedir = None
        self.iteration = None
        self.epoch = None
        self.best_psnr = None
        self.best_loss = None
        self.writer = None

        self.__setup()

    def __setup(self):


        self.basedir = join('checkpoints', self.opt.arch)
        if not os.path.exists(self.basedir):
            os.makedirs(self.basedir)

        self.best_psnr = 0
        self.best_loss = 1e6
        self.epoch = 0  # start from epoch 0 or last checkpoint epoch
        self.iteration = 0

        cuda = not self.opt.no_cuda
        self.device = 'cuda' if cuda else 'cpu'
        print('Cuda Acess: %d' % cuda)
        if cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        torch.manual_seed(self.opt.seed)
        if cuda:
            torch.cuda.manual_seed(self.opt.seed)

        """Model"""
        print("=> creating model '{}'".format(self.opt.arch))
        self.net = models.__dict__[self.opt.arch]()
        # initialize parameters
        #print(self.net)
        init_params(self.net, init_type=self.opt.init) # disable for default initialization

        if len(self.opt.gpu_ids) > 1:
            self.net  = nn.DataParallel(self.net.cuda(), device_ids=self.opt.gpu_ids, output_device=self.opt.gpu_ids[0])
        
        if self.opt.loss == 'l2':
            self.criterion = nn.MSELoss()
        if self.opt.loss == 'l1':
            self.criterion = nn.L1Loss()
        if self.opt.loss == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        if self.opt.loss == 'ssim':
            self.criterion = SSIMLoss(data_range=1, channel=31)
        if self.opt.loss == 'l2_ssim':
            self.criterion = MultipleLoss([nn.MSELoss(), SSIMLoss(data_range=1, channel=31)], weight=[1, 2.5e-3])
        if self.opt.loss == 'l2_sam':
            self.criterion = MultipleLoss([nn.MSELoss(),SAMLoss()],weight=[1, 1e-3])
        if self.opt.loss == 'cons':
            self.criterion = L1Consist([nn.L1Loss(),ContrastLoss(ablation=False)],weight=[1, 1])
        if self.opt.loss == 'cons_l2':
            self.criterion = L1Consist([nn.MSELoss(),ContrastLoss(ablation=False)],weight=[1, 0.01])

        print(self.criterion)

        if cuda:
            self.net.to(self.device)
            print('cuda initialized')
            self.criterion = self.criterion.to(self.device)

        """Logger Setup"""
        log = not self.opt.no_log
        if log:
            self.writer = get_summary_writer(os.path.join(self.basedir, 'logs'), self.opt.prefix)

        """Optimization Setup"""
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=self.opt.lr, weight_decay=self.opt.wd, amsgrad=False)

        """Resume previous model"""
        if self.opt.resume:
            # Load checkpoint.
            self.load(self.opt.resumePath, not self.opt.no_ropt)
        else:
            print('==> Building model..')
           # print(self.net)
        total = sum([param.nelement() for param in self.net.parameters()])    
        print("Number of parameter: %.2fM" % (total/1e6))
        

    #    # stat(self.net, (31, 64, 64))
        # from ptflops import get_model_complexity_info
        # if self.get_net().use_2dconv == True:
        #     macs, params = get_model_complexity_info(self.net,  (31, 512, 512),as_strings=True,
        #                                    print_per_layer_stat=False, verbose=False)
        # else:
        #      macs, params = get_model_complexity_info(self.net,  (1,31, 512, 512),as_strings=True,
        #                                    print_per_layer_stat=False, verbose=False)
        # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# #        print(self.net.flops([64,64]))
        # input_res= (31, 64, 64)

        # batch = torch.ones(()).new_empty((1, *input_res),
        #                                      dtype=next(self.net.parameters()).dtype,
        #                                      device=next(self.net.parameters()).device)
        # #print(input_res.shape)
        # #from fvcore.nn import FlopCountAnalysis
        # from flop_count.flop_count import FlopCountAnalysis
        # flops = FlopCountAnalysis(self.net, batch)
        # print(flops.total())

        # from thop import profile
        # batch = torch.randn(1,31, 512, 512)
        # macs, params = profile(self.net, inputs=(batch.to('cuda'), ))
        # print(macs,params)
       


        

        # from torchstat import stat
        # stat(self.net, (3, 256, 256))
#        print(self.net.flops([64,64]))

    def reset_params(self):
        init_params(self.net, init_type=self.opt.init) # disable for default initialization

    def forward(self, inputs):        
        if self.opt.chop:            
            output = self.forward_chop(inputs)
        else:
            output = self.net(inputs)
        
        return output

    def forward_chop(self, x, base=16):        
        n, c, b, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        shave_h = np.ceil(h_half / base) * base - h_half
        shave_w = np.ceil(w_half / base) * base - w_half

        shave_h = shave_h if shave_h >= 10 else shave_h + base
        shave_w = shave_w if shave_w >= 10 else shave_w + base

        h_size, w_size = int(h_half + shave_h), int(w_half + shave_w)        
        
        inputs = [
            x[..., 0:h_size, 0:w_size],
            x[..., 0:h_size, (w - w_size):w],
            x[..., (h - h_size):h, 0:w_size],
            x[..., (h - h_size):h, (w - w_size):w]
        ]

        outputs = [self.net(input_i) for input_i in inputs]

        output = torch.zeros_like(x)
        output_w = torch.zeros_like(x)
        
        output[..., 0:h_half, 0:w_half] += outputs[0][..., 0:h_half, 0:w_half]
        output_w[..., 0:h_half, 0:w_half] += 1
        output[..., 0:h_half, w_half:w] += outputs[1][..., 0:h_half, (w_size - w + w_half):w_size]
        output_w[..., 0:h_half, w_half:w] += 1
        output[..., h_half:h, 0:w_half] += outputs[2][..., (h_size - h + h_half):h_size, 0:w_half]
        output_w[..., h_half:h, 0:w_half] += 1
        output[..., h_half:h, w_half:w] += outputs[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
        output_w[..., h_half:h, w_half:w] += 1
        
        output /= output_w

        return output

    def __step(self, train, inputs, targets,sigma=None):        
        if train:
            self.optimizer.zero_grad()
        loss_data = 0
        total_norm = None
        self.net.eval()
        
        if self.get_net().bandwise:
            O = []
            for time, (i, t) in enumerate(zip(inputs.split(1, 1), targets.split(1, 1))):
                o = self.net(i)
                O.append(o)
                loss = self.criterion(o, t)
                if train:
                    loss.backward()
                loss_data += loss.item()
            outputs = torch.cat(O, dim=1)
        else:
           
            #noisy_sigma = torch.zeros
            
            outputs = self.net(inputs)
           # outputs = torch.clamp(outputs, 0, 1)
            # loss = self.criterion(outputs, targets)
            
            # if outputs.ndimension() == 5:
            #     loss = self.criterion(outputs[:,0,...], torch.clamp(targets[:,0,...], 0, 1))
            # else:
            #     loss = self.criterion(outputs, torch.clamp(targets, 0, 1))
            #print(outputs.shape,torch.squeeze(outputs).shape,targets.shape)
            #loss = self.criterion(outputs[:,0,...], targets[:,0,...])
            # if self.net.use_2dconv == True:
            #     loss = self.criterion(outputs[:,0,...], targets[:,0,...])
            # else:
            loss = self.criterion(outputs[...], targets) #memnet

            if train:
                loss.backward()
            loss_data += loss.item()
        if train:
            total_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.opt.clip)
            self.optimizer.step()

        return outputs, loss_data, total_norm



    def load(self, resumePath=None, load_opt=True):

        print('==> Resuming from checkpoint %s..' % resumePath)
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(resumePath )
        # if load_opt:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.get_net().load_state_dict(checkpoint['net'])

        

    def train(self, train_loader,val):
        print('\nEpoch: %d' % self.epoch)
        self.net.train()
        train_loss = 0
        train_psnr = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if not batch_idx%100:
                st = time.time()

            if not self.opt.no_cuda:
                inputs, targets = inputs.to(self.device), targets.to(self.device)        
                #print(inputs.shape,inputs.type)
            outputs, loss_data, total_norm = self.__step(True, inputs, targets)
            train_loss += loss_data
            avg_loss = train_loss / (batch_idx+1)
            psnr = np.mean(cal_bwpsnr(outputs, targets))
            train_psnr += psnr
            avg_psnr = train_psnr/ (batch_idx+1)
            if not self.opt.no_log:
                # wandb.log({'train_psnr':avg_psnr},step=self.iteration)
                # wandb.log({'train_loss':loss_data},step=self.iteration)
                # wandb.log({'train_avg_loss':avg_loss},step=self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_psnr'), avg_psnr, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_loss'), loss_data, self.iteration)
                self.writer.add_scalar(
                    join(self.prefix, 'train_avg_loss'), avg_loss, self.iteration)

            self.iteration += 1

            progress_bar(batch_idx, len(train_loader), 'AvgLoss: %.4e | Loss: %.4e | Norm: %.4e | Psnr: %4e' 
                         % (avg_loss, loss_data, total_norm,psnr))

            if not batch_idx % 100:
                ed=time.time()
                print(batch_idx,'/',len(train_loader),'  ',ed-st)


        self.epoch += 1
        if not self.opt.no_log:
            self.writer.add_scalar(
                join(self.prefix, 'train_loss_epoch'), avg_loss, self.epoch)

  
    def test(self, valid_loader, filen,save_path = None):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        if os.path.exists(filen):
            filenames = [
                    fn
                    for fn in os.listdir(filen)
                    if fn.endswith('.mat')
                ]
        print('[i] Eval dataset ...')

        
        print(len(valid_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.to(self.device), targets.to(self.device) 
                   
                outputs, loss_data, _ = self.__step(False, inputs, targets)
                psnr = np.mean(cal_bwpsnr(outputs, targets))
                sam = cal_sam(outputs, targets)
                #outputs = torch.clamp(self.net(inputs), 0, 1)
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                          % (avg_loss, psnr, avg_psnr))
                
                psnr = []
                h,w=inputs.shape[-2:]
                band = inputs.shape[-3]
                result = outputs.squeeze().cpu().detach().numpy()
            
                img = targets.squeeze().cpu().detach().numpy()
                
                for k in range(band):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR.append(sum(psnr)/len(psnr))
                
                mse = sum(sum(sum((result-img)**2)))
                mse /= band*h*w
                mse *= 255*255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(band):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM.append(sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                #print(np.arccos(temp)*180/np.pi)
                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(band):
                    ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                ergas = 100*np.sqrt(ergas/band)
                ERGAS.append(ergas)

                ##官方原本注释的部分
                if save_path is None:
                    save_path=self.opt.resultdir


                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for band in range(31):
                    img = result[band]*255#

                    cv2.imwrite(os.path.join(save_path, filenames[batch_idx][:-4] +'band_'+str(band)+'.jpg'),cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                
                # scio.savemat('/data/HSI_Data/Hyperspectral_Project/Urban_cvpr2023/'+self.opt.arch+'urban.mat',{'result':result})
                # result = np.clip(result,0,1)
                # for band in range(100,105):
                #     img = result[band]*255#
                #     cv2.imwrite(os.path.join(save_path, self.opt.arch +'_band_'+str(band)+'.jpg'),cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                # color_img = np.concatenate([result[0][np.newaxis,:],result[105][np.newaxis,:],result[207][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # print(color_img.shape)
                # cv2.imwrite(os.path.join(save_path, self.opt.arch +'color.jpg'),cv2.cvtColor(color_img.astype(np.uint8),cv2.COLOR_RGB2BGR))
                # result = img

                # color_img = np.concatenate([result[9][np.newaxis,:],result[15][np.newaxis,:],result[28][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # print(color_img.shape)
                # cv2.imwrite(os.path.join(save_path, filenames[batch_idx][:-4] +'color.png'),cv2.cvtColor(color_img.astype(np.uint8),cv2.COLOR_RGB2BGR))


        print(sum(PSNR)/len(PSNR), sum(RMSE)/len(RMSE), sum(SSIM)/len(SSIM), sum(SAM)/len(SAM), sum(ERGAS)/len(ERGAS))
        print(PSNR)
        print(avg_psnr, avg_loss,avg_sam)

        return {'PSNR':sum(PSNR)/len(PSNR), 'RMSE':sum(RMSE)/len(RMSE), 'SSIM':sum(SSIM)/len(SSIM), 'SAM':sum(SAM)/len(SAM), 'ERGAS':sum(ERGAS)/len(ERGAS),'avg_psnr':avg_psnr, 'avg_loss':avg_loss,'avg_sam':avg_sam}

    def test_rotate(self, valid_loader, filen, save_path=None):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        if os.path.exists(filen):
            filenames = [
                fn
                for fn in os.listdir(filen)
                if fn.endswith('.mat')
            ]
        print('[i] Eval dataset ...')

        print(len(valid_loader))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                if not self.opt.no_cuda:

                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                def rotate_last_channel_to_first_and_average(input,copy_target):
                    # 获取通道数
                    batch, channels,height, width= input.shape
                    target=copy_target
                    outputs = []
                    loss=[]
                    psnr=[]

                    # 循环通道数次
                    step=1
                    for i in range(channels//step):
                        # 将最后一个通道放到第一个通道位置
                        rotated_input = torch.cat([input[:, -1*step:, :, :], input[:, :-1*step, :, :]], dim=1)
                        rotated_target = torch.cat([target[:, -1*step:, :, :],target[:, :-1*step, :, :]],dim=1)


                        # 将轮换后的输入传入模型
                        output, loss_data, _ = self.__step(False, rotated_input,rotated_target)

                        for j in range(i+1):
                            output = torch.cat([output[:, 1*step:, :, :], output[:, :1*step, :, :]], dim=1)

                        # 将输出保存
                        outputs.append(output)
                        psnr.append(cal_bwpsnr(output, copy_target))
                        loss.append(loss_data)

                        # 更新输入，将轮换后的输入作为下一次迭代的基础
                        input = rotated_input
                        target=rotated_target

                    # 将所有输出求平均
                    final_output = torch.mean(torch.stack(outputs), dim=0)
                    final_loss=np.mean(loss)

                    return final_output,final_loss

                # 假设 input 是你的输入张量， model 是你现有的模型
                # 调用该函数
                outputs, loss_data= rotate_last_channel_to_first_and_average(inputs,targets)

                # outputs, loss_data, _ = self.__step(False, inputs, targets)
                psnr = np.mean(cal_bwpsnr(outputs, targets))
                sam = cal_sam(outputs, targets)
                # outputs = torch.clamp(self.net(inputs), 0, 1)
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx + 1)
                avg_sam = total_sam / (batch_idx + 1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx + 1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                             % (avg_loss, psnr, avg_psnr))

                psnr = []
                h, w = inputs.shape[-2:]
                band = inputs.shape[-3]
                result = outputs.squeeze().cpu().detach().numpy()

                img = targets.squeeze().cpu().detach().numpy()

                for k in range(band):
                    psnr.append(10 * np.log10((h * w) / sum(sum((result[k] - img[k]) ** 2))))
                PSNR.append(sum(psnr) / len(psnr))

                mse = sum(sum(sum((result - img) ** 2)))
                mse /= band * h * w
                mse *= 255 * 255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(band):
                    ssim.append((2 * np.mean(result[k]) * np.mean(img[k]) + k1 ** 2) \
                                * (2 * np.cov(result[k].reshape(h * w), img[k].reshape(h * w))[0, 1] + k2 ** 2) \
                                / (np.mean(result[k]) ** 2 + np.mean(img[k]) ** 2 + k1 ** 2) \
                                / (np.var(result[k]) + np.var(img[k]) + k2 ** 2))
                SSIM.append(sum(ssim) / len(ssim))

                temp = (np.sum(result * img, 0) + np.spacing(1)) \
                       / (np.sqrt(np.sum(result ** 2, 0) + np.spacing(1))) \
                       / (np.sqrt(np.sum(img ** 2, 0) + np.spacing(1)))
                # print(np.arccos(temp)*180/np.pi)
                sam = np.mean(np.arccos(temp)) * 180 / np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(band):
                    ergas += np.mean((img[k] - result[k]) ** 2) / np.mean(img[k]) ** 2
                ergas = 100 * np.sqrt(ergas / band)
                ERGAS.append(ergas)

                ##官方原本注释的部分
                if save_path is None:
                    save_path = self.opt.resultdir

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for band in range(31):
                    img = result[band] * 255  #

                    cv2.imwrite(os.path.join(save_path, filenames[batch_idx][:-4] + 'band_' + str(band) + '.jpg'),
                                cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))

                # scio.savemat('/data/HSI_Data/Hyperspectral_Project/Urban_cvpr2023/'+self.opt.arch+'urban.mat',{'result':result})
                # result = np.clip(result,0,1)
                # for band in range(100,105):
                #     img = result[band]*255#
                #     cv2.imwrite(os.path.join(save_path, self.opt.arch +'_band_'+str(band)+'.jpg'),cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                # color_img = np.concatenate([result[0][np.newaxis,:],result[105][np.newaxis,:],result[207][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # print(color_img.shape)
                # cv2.imwrite(os.path.join(save_path, self.opt.arch +'color.jpg'),cv2.cvtColor(color_img.astype(np.uint8),cv2.COLOR_RGB2BGR))
                # result = img

                # color_img = np.concatenate([result[9][np.newaxis,:],result[15][np.newaxis,:],result[28][np.newaxis,:]],0)
                # color_img = color_img.transpose((1,2,0))*255
                # print(color_img.shape)
                # cv2.imwrite(os.path.join(save_path, filenames[batch_idx][:-4] +'color.png'),cv2.cvtColor(color_img.astype(np.uint8),cv2.COLOR_RGB2BGR))

        print(sum(PSNR) / len(PSNR), sum(RMSE) / len(RMSE), sum(SSIM) / len(SSIM), sum(SAM) / len(SAM),
              sum(ERGAS) / len(ERGAS))
        print(PSNR)
        print(avg_psnr, avg_loss, avg_sam)

        return {'PSNR': sum(PSNR) / len(PSNR), 'RMSE': sum(RMSE) / len(RMSE), 'SSIM': sum(SSIM) / len(SSIM),
                'SAM': sum(SAM) / len(SAM), 'ERGAS': sum(ERGAS) / len(ERGAS), 'avg_psnr': avg_psnr,
                'avg_loss': avg_loss, 'avg_sam': avg_sam}

    def test_patch(self, valid_loader, filen,patch_size=64):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        filenames = [
                fn
                for fn in os.listdir(filen)
                if fn.endswith('.mat')
            ]
        print('[i] Eval dataset ...')
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                _,channel, width, height = inputs.shape
                input_patch = torch.zeros((64,31,64,64),dtype=torch.float)
                targets_patch = torch.zeros((64,31,64,64),dtype=torch.float)
                num = 0
                for i in range(width//patch_size):
                    for j in range(height//patch_size):
                        
                        sub_image = inputs[:,:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                        input_patch[num] = sub_image
                        targets_patch[num] = targets[:,:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                        num += 1
                if not self.opt.no_cuda:
                    inputs, targets = input_patch.to(self.device), targets_patch.to(self.device)   
                outputs, loss_data, _ = self.__step(False, inputs, targets)

                psnr = np.mean(cal_bwpsnr(outputs, targets))
                sam = cal_sam(outputs, targets)
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                          % (avg_loss, psnr, avg_psnr))
                
                psnr = []
                result_patch = outputs.squeeze().cpu().detach().numpy()
            
                img_patch = targets.squeeze().cpu().numpy()
                
                result = np.zeros((31,512,512))
                img = np.zeros((31,512,512))
                h,w=result.shape[-2:]
                num=0
                for i in range(width//patch_size):
                    for j in range(height//patch_size):
                        result[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = result_patch[num]
                        img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = img_patch[num]
                        num += 1
                for k in range(31):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR.append(sum(psnr)/len(psnr))
                
                mse = sum(sum(sum((result-img)**2)))
                mse /= 31*h*w
                mse *= 255*255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(31):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM.append(sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                #print(np.arccos(temp)*180/np.pi)
                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(31):
                    ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                ergas = 100*np.sqrt(ergas/31)
                ERGAS.append(ergas)
                
                # scio.savemat('/data/HSI_Data/Hyperspectral_Project/Urban_result/Ours/'+filenames[batch_idx], {'result': result})
        print(sum(PSNR)/len(PSNR), sum(RMSE)/len(RMSE), sum(SSIM)/len(SSIM), sum(SAM)/len(SAM), sum(ERGAS)/len(ERGAS))
        
        print(avg_psnr, avg_loss,avg_sam)
        return avg_psnr, avg_loss,avg_sam


    def test_3dpatch(self, valid_loader, filen,patch_size=64,band_size=31,all_size=512):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        filenames = [
                fn
                for fn in os.listdir(filen)
                if fn.endswith('.mat')
            ]
        print('[i] Eval dataset ...')
        blocks = (all_size//patch_size)*(all_size//patch_size)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                _,_,channel, width, height = inputs.shape
                input_patch = torch.zeros((blocks,band_size,patch_size,patch_size),dtype=torch.float)
                targets_patch = torch.zeros((blocks,band_size,patch_size,patch_size),dtype=torch.float)
                num = 0
                for i in range(width//patch_size):
                    for j in range(height//patch_size):
                        
                        sub_image = inputs[:,:,:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                        input_patch[num] = sub_image
                        targets_patch[num] = targets[:,:,:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                        num += 1
                if not self.opt.no_cuda:
                    inputs, targets = input_patch.to(self.device), targets_patch.to(self.device)  
                    inputs=inputs.unsqueeze(1) 
                outputs, loss_data, _ = self.__step(False, inputs, targets)

                psnr = np.mean(cal_bwpsnr(outputs, targets))
                sam = cal_sam(outputs, targets)
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                          % (avg_loss, psnr, avg_psnr))
                
                psnr = []
                result_patch = outputs.squeeze().cpu().detach().numpy()
                img_patch = targets.squeeze().cpu().numpy()
                
                result = np.zeros((band_size,all_size,all_size))
                img = np.zeros((band_size,all_size,all_size))
                h,w=result.shape[-2:]
                num=0
                for i in range(width//patch_size):
                    for j in range(height//patch_size):
                        result[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = result_patch[num]
                        img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = img_patch[num]
                        num += 1
                for k in range(band_size):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR.append(sum(psnr)/len(psnr))
                
                mse = sum(sum(sum((result-img)**2)))
                mse /= band_size*h*w
                mse *= 255*255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(band_size):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM.append(sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                #print(np.arccos(temp)*180/np.pi)
                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(band_size):
                    ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                ergas = 100*np.sqrt(ergas/band_size)
                ERGAS.append(ergas)
                
                # scio.savemat('/data/HSI_Data/Hyperspectral_Project/Urban_result/Ours/'+filenames[batch_idx], {'result': result})
        print(sum(PSNR)/len(PSNR), sum(RMSE)/len(RMSE), sum(SSIM)/len(SSIM), sum(SAM)/len(SAM), sum(ERGAS)/len(ERGAS))
        
        print(avg_psnr, avg_loss,avg_sam)
        return avg_psnr, avg_loss,avg_sam

    def validate(self, valid_loader, name,patch_size=64):
        self.net.eval()
        validate_loss = 0
        total_psnr = 0
        total_sam = 0
        RMSE = []
        SSIM = []
        SAM = []
        ERGAS = []
        PSNR = []
        print('[i] Eval dataset {}...'.format(name))
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                
                if ('cswin_unet' in self.opt.arch) or ('unfold' in self.opt.arch)or ('scalable' in self.opt.arch):
                    _,channel, width, height = inputs.shape
                    input_patch = torch.zeros((64,31,64,64),dtype=torch.float)                 
                    targets_patch = torch.zeros((64,31,64,64),dtype=torch.float)
                    num=0
                    for i in range(width//patch_size):                     
                        for j in range(height//patch_size):                                                 
                            sub_image = inputs[:,:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]                         
                            input_patch[num] = sub_image                   
                            targets_patch[num] = targets[:,:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]                         
                            num += 1
                    if not self.opt.no_cuda:                     
                        inputs, targets = input_patch.to(self.device), targets_patch.to(self.device)   
                else:                 
                    inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs, loss_data, _ = self.__step(False, inputs, targets)
                psnr = np.mean(cal_bwpsnr(outputs, targets))
                sam = cal_sam(outputs, targets)
                validate_loss += loss_data
                total_sam += sam
                avg_loss = validate_loss / (batch_idx+1)
                avg_sam = total_sam / (batch_idx+1)

                total_psnr += psnr
                avg_psnr = total_psnr / (batch_idx+1)

                progress_bar(batch_idx, len(valid_loader), 'Loss: %.4e | PSNR: %.4f | AVGPSNR: %.4f '
                          % (avg_loss, psnr, avg_psnr))
                
                psnr = []
                h,w=inputs.shape[-2:]
                if ('cswin_unet' in self.opt.arch) or ('unfold' in self.opt.arch) or('scalable' in self.opt.arch):
                    result_patch = outputs.squeeze().cpu().detach().numpy()
            
                    img_patch = targets.squeeze().cpu().numpy()
                
                    result = np.zeros((31,512,512))
                    img = np.zeros((31,512,512))
                    h,w=result.shape[-2:]
                    num=0
                    for i in range(width//patch_size):
                        for j in range(height//patch_size):
                            result[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = result_patch[num]
                            img[:, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = img_patch[num]
                            num += 1
                else:
                   # outputs = torch.clamp(outputs,0,1) 
                    result = outputs.squeeze().cpu().detach().numpy()
            
                    img = targets.squeeze().cpu().numpy()
                for k in range(31):
                    psnr.append(10*np.log10((h*w)/sum(sum((result[k]-img[k])**2))))
                PSNR.append(sum(psnr)/len(psnr))
                
                mse = sum(sum(sum((result-img)**2)))
                mse /= 31*h*w
                mse *= 255*255
                rmse = np.sqrt(mse)
                RMSE.append(rmse)

                ssim = []
                k1 = 0.01
                k2 = 0.03
                for k in range(31):
                    ssim.append((2*np.mean(result[k])*np.mean(img[k])+k1**2) \
                        *(2*np.cov(result[k].reshape(h*w), img[k].reshape(h*w))[0,1]+k2**2) \
                        /(np.mean(result[k])**2+np.mean(img[k])**2+k1**2) \
                        /(np.var(result[k])+np.var(img[k])+k2**2))
                SSIM.append(sum(ssim)/len(ssim))

                temp = (np.sum(result*img, 0) + np.spacing(1)) \
                    /(np.sqrt(np.sum(result**2, 0) + np.spacing(1))) \
                    /(np.sqrt(np.sum(img**2, 0) + np.spacing(1)))
                #print(np.arccos(temp)*180/np.pi)
                sam = np.mean(np.arccos(temp))*180/np.pi
                SAM.append(sam)

                ergas = 0.
                for k in range(31):
                    ergas += np.mean((img[k]-result[k])**2)/np.mean(img[k])**2
                ergas = 100*np.sqrt(ergas/31)
                ERGAS.append(ergas)
                
        
        print(sum(PSNR)/len(PSNR), sum(RMSE)/len(RMSE), sum(SSIM)/len(SSIM), sum(SAM)/len(SAM), sum(ERGAS)/len(ERGAS))
        if not self.opt.no_log:
            # wandb.log({'val_loss_epoch':avg_loss,'val_psnr_epoch':avg_psnr,'val_sam_epoch':avg_sam,'epoch':self.epoch})
            
            self.writer.add_scalar(
                join(self.prefix, name, 'val_loss_epoch'), avg_loss, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_psnr_epoch'), avg_psnr, self.epoch)
            self.writer.add_scalar(
                join(self.prefix, name, 'val_sam_epoch'), avg_sam, self.epoch)


        print(avg_psnr, avg_loss,avg_sam)
        return avg_psnr, avg_loss,avg_sam


  
    def save_checkpoint(self, model_out_path=None, **kwargs):
        if not model_out_path:
            model_out_path = join(self.basedir, self.prefix, "model_epoch_%d_%d.pth" % (
                self.epoch, self.iteration))

        state = {
            'net': self.get_net().state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'iteration': self.iteration,
        }
        
        state.update(kwargs)

        if not os.path.isdir(join(self.basedir, self.prefix)):
            os.makedirs(join(self.basedir, self.prefix))

        torch.save(state, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    # saving result into disk
    def test_develop(self, test_loader, savedir=None, verbose=True):
        from scipy.io import savemat
        from os.path import basename, exists

        def torch2numpy(hsi):
            if self.net.use_2dconv:
                R_hsi = hsi.data[0].cpu().numpy().transpose((1,2,0))
            else:
                R_hsi = hsi.data[0].cpu().numpy()[0,...].transpose((1,2,0))
            return R_hsi    

        self.net.eval()
        test_loss = 0
        total_psnr = 0
        dataset = test_loader.dataset.dataset

        res_arr = np.zeros((len(test_loader), 3))
        input_arr = np.zeros((len(test_loader), 3))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs, loss_data, _ = self.__step(False, inputs, targets)
                
                test_loss += loss_data
                avg_loss = test_loss / (batch_idx+1)
                
                res_arr[batch_idx, :] = MSIQA(outputs, targets)
                input_arr[batch_idx, :] = MSIQA(inputs, targets)

                """Visualization"""
                # Visualize3D(inputs.data[0].cpu().numpy())
                # Visualize3D(outputs.data[0].cpu().numpy())

                psnr = res_arr[batch_idx, 0]
                ssim = res_arr[batch_idx, 1]
                if verbose:
                    print(batch_idx, psnr, ssim)

                if savedir:
                    filedir = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0])  
                    outpath = join(filedir, '{}.mat'.format(self.opt.arch))

                    if not exists(filedir):
                        os.mkdir(filedir)

                    if not exists(outpath):
                        savemat(outpath, {'R_hsi': torch2numpy(outputs)})
                        
        return res_arr, input_arr

    def test_real(self, test_loader, savedir=None):
        """Warning: this code is not compatible with bandwise flag"""
        from scipy.io import savemat
        from os.path import basename
        self.net.eval()
        dataset = test_loader.dataset.dataset

        with torch.no_grad():
            for batch_idx, inputs in enumerate(test_loader):
                if not self.opt.no_cuda:
                    inputs = inputs.cuda()           

                outputs = self.forward(inputs)

                """Visualization"""                
                input_np = inputs[0].cpu().numpy()
                output_np = outputs[0].cpu().numpy()

                display = np.concatenate([input_np, output_np], axis=-1)
                
                Visualize3D(display)
                # Visualize3D(outputs[0].cpu().numpy())
                # Visualize3D((outputs-inputs).data[0].cpu().numpy())
                
                if savedir:
                    R_hsi = outputs.data[0].cpu().numpy()[0,...].transpose((1,2,0))     
                    savepath = join(savedir, basename(dataset.filenames[batch_idx]).split('.')[0], self.opt.arch + '.mat')
                    savemat(savepath, {'R_hsi': R_hsi})
        
        return outputs

    def get_net(self):
        if len(self.opt.gpu_ids) > 1:
            return self.net.module
        else:
            return self.net           
