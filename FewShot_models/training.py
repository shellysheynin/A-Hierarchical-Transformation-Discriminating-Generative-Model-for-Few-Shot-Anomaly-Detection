import FewShot_models.functions as functions
import FewShot_models.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from FewShot_models.imresize import imresize
from FewShot_models.augments_transform_color import apply_transform
import shutil
import torch.utils.data
import torch.utils.data
from FewShot_models.manipulate import *


def train(opt,Gs,Zs,reals,NoiseAmp):

    reals, Zs, NoiseAmp, in_s, scale_num = functions.collect_reals(opt, reals, Zs, NoiseAmp)
    nfc_prev = 0
    for index_image in range(int(opt.num_images)):
        NoiseAmp[index_image] = []

    while scale_num < opt.stop_scale+1:

        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        D_curr, G_curr = init_models(opt)

        index_arr_flag = 0

        for epoch in range(opt.num_epochs):

            optimizerD = optim.Adam(D_curr.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
            optimizerG = optim.Adam(G_curr.parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999))
            schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[500], gamma=opt.gamma)
            schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[1000], gamma=opt.gamma)

            print(" ")
            print("this is class: ", opt.pos_class)
            print("size image: ", opt.size_image)
            print("num images: ", opt.num_images)
            print(" this is the noise amp for this image in this scale: ", opt.noise_amp)
            print("index of download: ", opt.index_download)
            print("min size image: ", opt.min_size)
            print("max size image: ", opt.max_size)
            print("num layers: ", opt.num_layer)
            print("num transforms: ", opt.num_transforms)

            print(" ")
            index_image = range(int(opt.num_images))


            G_curr = functions.reset_grads(G_curr, True)
            D_curr = functions.reset_grads(D_curr, True)

            opt.out_ = functions.generate_dir2save(opt)
            opt.global_outf = '%s/%d' % (opt.out_, scale_num)
            if os.path.exists(opt.global_outf):
                shutil.rmtree( opt.global_outf)

            opt.outf = ['%s/%d/index_image_%d' % (opt.out_, scale_num, id) for id in index_image]
            try:
                for j in opt.outf:
                    try:
                        os.makedirs(j)
                    except:
                        pass
            except OSError as err:
                print("OS error: {0}".format(err))

                pass

            for id in index_image:
                plt.imsave('%s/real_scale%d.png' % (opt.outf[id], id), functions.convert_image_np(reals[id][scale_num]), vmin=0, vmax=1)

            if (nfc_prev==opt.nfc):

                G_curr.load_state_dict(torch.load('%s/%d/netG.pth' % (opt.out_,scale_num-1)))
                D_curr.load_state_dict(torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1)))
                nfc_prev = 0

            if not index_arr_flag:
                real = torch.cat([reals[id][scale_num] for id in index_image], dim=0)
                opt.nzx = real.shape[2]  # +(opt.ker_size-1)*(opt.num_layer)
                opt.nzy = real.shape[3]  # +(opt.ker_size-1)*(opt.num_layer)
                opt.receptive_field = opt.ker_size + ((opt.ker_size - 1) * (opt.num_layer - 1)) * opt.stride  # 11
                pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)  # 5
                m_noise = nn.ZeroPad2d(int(pad_noise))
                z_opt = functions.generate_noise([1, opt.nzx, opt.nzy],device=opt.device)
                z_opt = m_noise(z_opt.expand(real.shape[0], 3, opt.nzx, opt.nzy))

            elif index_arr_flag:
                z_opt = torch.cat(([Zs[id][scale_num] for id in index_image]), dim=0)

            in_s, G_curr = train_single_scale(D_curr,G_curr,reals,Gs,Zs,in_s,NoiseAmp,opt, index_image, z_opt,
                                             optimizerD, optimizerG, schedulerD, schedulerG, index_arr_flag)

            if not index_arr_flag:
                print(" ")
                print("this is class: ", opt.pos_class)
                print("size image: ", opt.size_image)
                print("num images: ", opt.num_images)
                print(" this is the noise amp for this image in this scale: ", opt.noise_amp)
                print("index of download: ", opt.index_download)
                print("min size image: ", opt.min_size)
                print("max size image: ", opt.max_size)
                print("num layers: ", opt.num_layer)
                print("num transforms: ", opt.num_transforms)

                print(" ")
                for id in index_image:
                    Zs[id].append(z_opt[id:id+1])
                    NoiseAmp[id].append(opt.noise_amp[id:id+1])
            else:
                for id in index_image:
                    NoiseAmp[id][scale_num] = opt.noise_amp[id:id+1]

            index_arr_flag = True

            G_curr = functions.reset_grads(G_curr,False)
            G_curr.eval()
            D_curr = functions.reset_grads(D_curr,False)
            D_curr.eval()

        Gs.append(G_curr)
        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr, optimizerD, optimizerG
        torch.cuda.empty_cache()

    return

def pad_image_id(real,  index_image):
    id_padding = [torch.full((1, 1, real.shape[2], real.shape[3]), id, dtype=torch.float).cuda() for id in
                  index_image]
    id_padding = torch.cat(id_padding, dim=0)

    padded_id_real = torch.cat((real, id_padding), 1)
    return padded_id_real


def apply_augmentation(real, is_flip, tx, ty, k_rotate,flag_color):
    try:
        augment = apply_transform(real, is_flip, tx, ty, k_rotate,flag_color)
    except:
        augment = real


    return augment

def get_err_D_fake_and_backward(output, num_transforms):

    loss_fn = nn.CrossEntropyLoss()
    reshaped_output = output.permute(0, 2, 3, 1).contiguous()
    shape = reshaped_output.shape
    reshaped_output = reshaped_output.view(-1,  shape[3])
    target = torch.ones(reshaped_output.shape[0])*(num_transforms)
    target = target.cuda().type(torch.cuda.LongTensor)
    loss = loss_fn(reshaped_output, target)
    loss.backward(retain_graph=True)
    return loss

def get_err_D_real_and_backward(output, ids_target):
    loss_fn = nn.CrossEntropyLoss()
    reshaped_output = output.permute(0, 2, 3, 1).contiguous()
    shape = reshaped_output.shape
    reshaped_output = reshaped_output.view(-1,  shape[3])
    target = torch.stack(ids_target).reshape(-1).type(torch.cuda.LongTensor)
    loss = loss_fn(reshaped_output, target)
    loss.backward(retain_graph=True)

    return loss

def get_err_G_fake_and_backward(output, ids_target):
    loss_fn = nn.CrossEntropyLoss()
    reshaped_output = output.permute(0, 2, 3, 1).contiguous()
    shape = reshaped_output.shape
    reshaped_output = reshaped_output.view(-1,  shape[3])
    target = torch.stack(ids_target).reshape(-1).type(torch.cuda.LongTensor)
    loss = loss_fn(reshaped_output, target)
    loss.backward(retain_graph=True)
    return loss

def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt, index_image, z_opt, optimizerD, optimizerG, schedulerD, schedulerG, is_passing_im_before, centers=None):

    real = torch.cat([reals[id][len(Gs)] for id in index_image], dim=0)
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))
    alpha = opt.alpha

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    z_opt2plot = []
    flag_first_time = True

    for epoch in range(opt.niter):

        if (Gs == []) & (opt.mode != 'SR_train'): # the beginning of pyramid and this is not super resolution
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device, num_samp=real.shape[0])
            noise_ = m_noise(noise_.expand(real.shape[0],3,opt.nzx,opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device, num_samp=real.shape[0])
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################

        for p in netD.parameters():
            p.requires_grad = True  # to avoid computation

        for j in range(opt.Dsteps):


            netD.zero_grad()
            reals_arr = []
            num_transforms = 0

            for index_transform, pair in enumerate(opt.list_transformations):

                num_transforms +=1
                flag_color,is_flip, tx, ty, k_rotate = pair
                real_transform = apply_augmentation(real, is_flip, tx, ty, k_rotate,flag_color).cuda()
                real_transform = torch.squeeze(real_transform)
                reals_arr.append(real_transform)
            opt.num_transforms = num_transforms
            num_transforms_range = range(num_transforms)
            real_transform = torch.stack(reals_arr)
            output = netD(real_transform).to(opt.device)
            id_padding = [torch.full((1, 1, output.shape[2], output.shape[3]), id, dtype=torch.float).cuda() for id in num_transforms_range]
            errD_real = get_err_D_real_and_backward(output, id_padding)
            D_x = -errD_real.item()

            # train with fake - this is the first time in this scale
            if (j==0) & (epoch == 0):

                if (Gs == [])  & (is_passing_im_before == False):
                    prev = torch.full([real.shape[0],opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device, dtype=torch.long)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([real.shape[0],opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device, dtype=torch.long)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = torch.full([real.shape[0], 1], 1, dtype=torch.long).cuda()
                    opt.noise_amp_tensor = torch.full([real.shape[0], opt.nc_z, opt.nzx, opt.nzy], 1,
                                                      dtype=torch.long).cuda()
                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt,index_image, index_transform,  is_flip, tx, ty, k_rotate)
                    prev = m_image(prev)

                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt,index_image, index_transform,  is_flip, tx, ty, k_rotate)
                    criterion = nn.MSELoss(reduction='none')
                    opt.noise_amp = torch.cat(([NoiseAmp[id][len(Gs) - 1] for id in range(opt.num_images)]),
                                              dim=0).cuda()

                    if not is_passing_im_before:
                        first_loss = criterion(real, z_prev)
                        RMSE = torch.sqrt(first_loss)
                        opt.noise_amp_init_tensor = torch.full([real.shape[0], opt.nc_z, opt.nzx, opt.nzy],
                                                               opt.noise_amp_init,
                                                               dtype=torch.float).cuda()
                        opt.noise_amp = opt.noise_amp_init_tensor * RMSE
                        opt.noise_amp = torch.unsqueeze(
                            torch.mean(torch.flatten(opt.noise_amp, start_dim=1), dim=1), dim=1)
                        opt.noise_amp_tensor = torch.full([1, opt.nc_z, opt.nzx, opt.nzy],
                                                          opt.noise_amp[0][0].item(), dtype=torch.float).cuda()
                        for i in range(1, real.shape[0]):
                            noise_tensor_temp = torch.full([1, opt.nc_z, opt.nzx, opt.nzy],
                                              opt.noise_amp[i][0].item(), dtype=torch.float).cuda()
                            opt.noise_amp_tensor = torch.cat((opt.noise_amp_tensor, noise_tensor_temp), dim=0)

                    else:
                        opt.noise_amp = torch.cat(([NoiseAmp[id][len(Gs) - 1] for id in range(opt.num_images)]),
                                                  dim=0).cuda()
                        opt.noise_amp_tensor = torch.full([1, opt.nc_z, opt.nzx, opt.nzy],
                                                          opt.noise_amp[0][0].item(), dtype=torch.float).cuda()
                        for i in range(1, real.shape[0]):
                            noise_tensor_temp = torch.full([1, opt.nc_z, opt.nzx, opt.nzy],
                                              opt.noise_amp[i][0].item(), dtype=torch.float).cuda()
                            opt.noise_amp_tensor = torch.cat((opt.noise_amp_tensor, noise_tensor_temp), dim=0)
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt, index_image, index_transform,  is_flip, tx, ty, k_rotate)
                prev = m_image(prev)


            if Gs == []:
                noise = noise_
            else:
                if flag_first_time == True:
                    flag_first_time=False
                noise = m_noise(opt.noise_amp_tensor)*noise_+prev

            padded_id_noise = pad_image_id(noise, index_image)
            fake = netG(padded_id_noise.detach(),prev)
            fakes_arr = []
            for index_transform, pair in enumerate(opt.list_transformations):
                flag_color,is_flip, tx, ty, k_rotate = pair
                fake_transform = apply_augmentation(fake.detach(), is_flip, tx, ty, k_rotate,flag_color).cuda()
                fake_transform = torch.squeeze(fake_transform)
                fakes_arr.append(fake_transform)
            fake_transform = torch.stack(fakes_arr)
            output = netD(fake_transform).to(opt.device)
            errD_fake = get_err_D_fake_and_backward(output, opt.num_transforms)
            D_G_z = output.mean().item()
            errD = errD_real + errD_fake

            if j == opt.Dsteps -1 and index_transform == opt.num_transforms -1 and (epoch % 50 == 0 or epoch == (opt.niter-1)):
                for param_group in optimizerD.param_groups:
                    print("learning rate D: ", param_group['lr'])
                print("errD real: ", D_x, "errD fake: ", D_G_z)

            optimizerD.step()

        errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for p in netD.parameters():
            p.requires_grad = False

        for j in range(opt.Gsteps):
            netG.zero_grad()
            padded_id_noise = pad_image_id(noise, index_image)
            fake = netG(padded_id_noise, prev)
            fakes_arr_G = []
            for index_transform, pair in enumerate(opt.list_transformations):
                flag_color,is_flip, tx, ty, k_rotate = pair
                fake_transform_G = apply_augmentation(fake, is_flip, tx, ty, k_rotate,flag_color).cuda()
                fake_transform_G = torch.squeeze(fake_transform_G)
                fakes_arr_G.append(fake_transform_G)
            fake_transform_G = torch.stack(fakes_arr_G)
            output = netD(fake_transform_G)
            id_padding = [torch.full((1, 1, output.shape[2], output.shape[3]), id, dtype=torch.float).cuda() for id in num_transforms_range]
            errG = get_err_G_fake_and_backward(output, id_padding)

            if alpha!=0:
                loss = nn.MSELoss()
                Z_opt = m_noise(opt.noise_amp_tensor)*z_opt+z_prev
                padded_id_Z_opt = pad_image_id(Z_opt.detach(),index_image)
                rec_loss = alpha*loss(netG(padded_id_Z_opt.detach(), z_prev), real)
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()
            else:
                Z_opt = z_opt
                rec_loss = 0
            if j == opt.Gsteps -1 and index_transform == opt.num_transforms -1 and (epoch % 50 == 0 or epoch == (opt.niter-1)):
                print("errG fake: ", errG.detach().item(), "rec loss: ", rec_loss.detach().item())

            optimizerG.step()

        errG2plot.append(errG.detach()+rec_loss)
        D_real2plot.append(D_x)
        z_opt2plot.append(rec_loss)

        if index_transform == opt.num_transforms -1 and(epoch % 25 == 0 or epoch == (opt.niter-1)):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))
            print(" ")

        if index_transform == opt.num_transforms -1 and (epoch % 150 == 0 or epoch == (opt.niter - 1)):

            for j, id in enumerate(opt.outf):
                plt.imsave('%s/fake_sample_epoch%d.png' % (id,  epoch),
                           functions.convert_image_np(fake[j:j + 1].detach()), vmin=0, vmax=1)
            padded_id_Z_opt = pad_image_id(Z_opt.detach() , index_image)
            G_z_opt = netG(padded_id_Z_opt.detach(), z_prev.detach())
            for j, id in enumerate(opt.outf):
                plt.imsave('%s/G(z_opt)_epoch%d.png' % (id, epoch),
                           functions.convert_image_np(G_z_opt[j:j + 1].detach()), vmin=0, vmax=1)

            torch.save(z_opt, '%s/z_opt.pth' % (opt.global_outf))

        schedulerD.step()
        schedulerG.step()
    functions.save_networks(netG,netD,z_opt,opt)

    return in_s,netG


def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt,index_image, index_transform,  is_flip, tx, ty, k_rotate):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            for scale_idx, (G) in enumerate(Gs):
                Z_opt = torch.cat([Zs[idx][scale_idx] for idx in index_image], dim=0)
                real_curr = torch.cat([reals[idx][scale_idx] for idx in index_image], dim=0)
                real_next = torch.cat([reals[idx][1:][scale_idx] for idx in index_image], dim=0)
                noise_amp = torch.cat(([NoiseAmp[id][scale_idx] for id in range(opt.num_images)]), dim=0).cuda()
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device, num_samp=real_curr.shape[0])
                    z = z.expand(real_curr.shape[0], 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device, num_samp=real_curr.shape[0])

                noise_amp_tensor = torch.full([1, z.shape[1], z.shape[2], z.shape[3]], noise_amp[0][0].item(), dtype=torch.float).cuda()
                for i in range(1, opt.num_images):
                    noise_tensor_temp = torch.full([1, z.shape[1], z.shape[2], z.shape[3]],
                                      noise_amp[i][0].item(), dtype=torch.float).cuda()
                    noise_amp_tensor = torch.cat((noise_amp_tensor, noise_tensor_temp), dim=0)

                z = m_noise(z)
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = m_noise(noise_amp_tensor) * z + G_z
                padded_id_z_in = pad_image_id(z_in, index_image)
                G_z_temp = G(padded_id_z_in.detach(),G_z)
                G_z = imresize(torch.unsqueeze(G_z_temp[0], dim=0),1/opt.scale_factor,opt)
                for id in range(1,opt.num_images):
                    G_z = torch.cat((G_z,imresize(torch.unsqueeze(G_z_temp[id], dim=0),1/opt.scale_factor,opt)), dim=0)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1

        if mode == 'rec':
            count = 0
            for scale_idx, (G) in enumerate((Gs)):

                Z_opt = torch.cat([Zs[idx][scale_idx] for idx in index_image], dim=0)
                real_curr = torch.cat([reals[idx][scale_idx] for idx in index_image], dim=0)
                real_next = torch.cat([reals[idx][1:][scale_idx] for idx in index_image], dim=0)
                noise_amp = torch.cat(([NoiseAmp[id][scale_idx] for id in range(opt.num_images)]), dim=0).cuda()

                noise_amp_tensor = torch.full([1, Z_opt.shape[1], Z_opt.shape[2], Z_opt.shape[3]],
                                              noise_amp[0][0].item(), dtype=torch.float).cuda()
                for i in range(1, opt.num_images):
                    noise_tensor_temp = torch.full([1, Z_opt.shape[1], Z_opt.shape[2], Z_opt.shape[3]],
                                      noise_amp[i][0].item(), dtype=torch.float).cuda()
                    noise_amp_tensor = torch.cat((noise_amp_tensor, noise_tensor_temp), dim=0)

                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp_tensor*Z_opt+G_z
                padded_id_z_in = pad_image_id(z_in,index_image)
                G_z_temp = G(padded_id_z_in.detach(),G_z)
                G_z = imresize(torch.unsqueeze(G_z_temp[0], dim=0), 1 / opt.scale_factor, opt)

                for id in range(1, opt.num_images):
                    G_z = torch.cat((G_z, imresize(torch.unsqueeze(G_z_temp[id], dim=0), 1 / opt.scale_factor, opt)),
                                    dim=0)
                G_z = G_z[:, :, 0:real_next.shape[2], 0:real_next.shape[3]]
                count += 1

    return G_z

def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))

    #discriminator initialization:
    netD = models.WDiscriminatorMulti(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    return netD, netG