from __future__ import print_function
import FewShot_models.functions as functions
import FewShot_models.models
import argparse
import os
import random
from FewShot_models.imresize import imresize
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage import io as img
import numpy as np
from skimage import color
import math
import imageio
import matplotlib.pyplot as plt
from config_few_shot import get_arguments

def pad_image_id(real,  index_image):

    id_padding = [torch.full((1, 1, real.shape[2], real.shape[3]), id, dtype=torch.float).cuda() for id in
                  index_image]
    id_padding = torch.cat(id_padding, dim=0)

    padded_id_real = torch.cat((real, id_padding), 1)

    return padded_id_real


def SinGAN_generate(Gs,Zs,reals,NoiseAmp,opt,in_s=None,scale_v=1,scale_h=1,n=0,gen_start_scale=0,num_samples=200):

    if in_s is None:
        in_s = torch.full(reals[0][0].shape, 0, device=opt.device, dtype=torch.long)
    images_cur = []
    index_image = range(int(opt.num_images))

    for scale_idx, (G) in enumerate(Gs):
        Z_opt = torch.cat([Zs[idx][scale_idx] for idx in index_image], dim=0)
        noise_amp = torch.cat(([NoiseAmp[id][scale_idx] for id in range(opt.num_images)]), dim=0).cuda()
        pad1 = ((opt.ker_size-1)*opt.num_layer)/2
        m = nn.ZeroPad2d(int(pad1))
        nzx = (Z_opt.shape[2]-pad1*2)*scale_v
        nzy = (Z_opt.shape[3]-pad1*2)*scale_h
        images_prev = images_cur
        images_cur = []

        for i in range(0,num_samples,1):
            if n == 0:
                z_curr = functions.generate_noise([1,nzx,nzy], device=opt.device)
                z_curr = z_curr.expand(1,3,z_curr.shape[2],z_curr.shape[3])
                z_curr = m(z_curr)
            else:
                z_curr = functions.generate_noise([opt.nc_z,nzx,nzy], device=opt.device)
                z_curr = m(z_curr)

            if images_prev == []:
                I_prev = m(in_s)
            else:
                I_prev_temp = images_prev[i]
                I_prev = imresize(torch.unsqueeze(I_prev_temp[0], dim=0), 1 / opt.scale_factor, opt)
                for id in range(1, opt.num_images):
                    I_prev = torch.cat((I_prev, imresize(torch.unsqueeze(I_prev_temp[id], dim=0), 1 / opt.scale_factor, opt)),
                                    dim=0)
                if opt.mode != "SR":
                    I_prev = I_prev[:, :, 0:round(scale_v * reals[0][n].shape[2]), 0:round(scale_h * reals[0][n].shape[3])]
                    I_prev = m(I_prev)
                    I_prev = I_prev[:,:,0:z_curr.shape[2],0:z_curr.shape[3]]
                    I_prev = functions.upsampling(I_prev,z_curr.shape[2],z_curr.shape[3])
                else:
                    I_prev = m(I_prev)

            if n < gen_start_scale:
                z_curr = Z_opt

            noise_amp_tensor = torch.full([1, z_curr.shape[1], z_curr.shape[2], z_curr.shape[3]], noise_amp[0][0].item(),
                                          dtype=torch.float).cuda()
            for j in range(1, opt.num_images):
                temp = torch.full([1, z_curr.shape[1], z_curr.shape[2], z_curr.shape[3]],
                                  noise_amp[j][0].item(), dtype=torch.float).cuda()
                noise_amp_tensor = torch.cat((noise_amp_tensor, temp), dim=0)
            z_in = noise_amp_tensor*(z_curr)+I_prev
            padded_id_Z_opt = pad_image_id(z_in, index_image)


            I_curr = G(padded_id_Z_opt.detach(),I_prev)

            if n == len(Gs)-1:
                if opt.mode == 'train':
                    dir2save = '%s/RandomSamples/%s/gen_start_scale=%d' % (opt.out, opt.input_name[:-4], gen_start_scale)
                else:
                    dir2save = functions.generate_dir2save(opt)
                try:
                    os.makedirs(dir2save)
                except OSError:
                    pass
                if (opt.mode != "harmonization") & (opt.mode != "editing") & (opt.mode != "SR") & (opt.mode != "paint2image"):
                    for j in range(opt.num_images):

                        plt.imsave('%s/%d%d.png' % (dir2save, i,j), functions.convert_image_np(I_curr[j].unsqueeze(dim=0).detach()), vmin=0,vmax=1)
            images_cur.append(I_curr)
        n+=1
    return I_curr.detach()
