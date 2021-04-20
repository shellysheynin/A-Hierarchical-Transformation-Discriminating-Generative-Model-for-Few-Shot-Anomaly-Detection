import torch
import torch.nn.functional as F
from FewShot_models.manipulate import *
from kornia.color import *
import kornia.augmentation.functional as F_k
import kornia as K
from kornia.geometry.transform.imgwarp import (
    warp_affine, get_rotation_matrix2d, get_affine_matrix2d
)
import itertools
import cv2
import kornia
def apply_transform(x, is_flip, tx, ty, k_rotate, flag_color, channels_first=True):
    if not channels_first:
        x = x.permute(0, 3, 1, 2)
    if is_flip == True:
        x = h_flip(x, is_flip)
    if tx != 0 or ty != 0:
        x = translation(x, tx, ty)
    if k_rotate != 0:
        x = rotate_90(x,k_rotate)
    if not channels_first:
        x = x.permute(0, 2, 3, 1)
    if flag_color != 0:
        x = color_transform(x,flag_color)
    x = x.contiguous()
    return x

def color_transform(x, flag):
    if flag == 0: #rgb
        return x
    elif flag == 1: #gray
        # x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = kornia.rgb_to_grayscale(x)
        x = x.repeat(1, 3, 1,1)
    return x
def affine(tensor: torch.Tensor, matrix: torch.Tensor, mode: str = 'bilinear',
           align_corners: bool = False) -> torch.Tensor:

    # warping needs data in the shape of BCHW
    is_unbatched = tensor.ndimension() == 3
    if is_unbatched:
        tensor = torch.unsqueeze(tensor, dim=0)

    # we enforce broadcasting since by default grid_sample it does not
    # give support for that
    matrix = matrix.expand(tensor.shape[0], -1, -1)

    # warp the input tensor
    height = tensor.shape[-2]
    width = tensor.shape[-1]
    warped = warp_affine(tensor, matrix, (height, width), mode, padding_mode = 'reflection',
                                       align_corners=align_corners)

    # return in the original shape
    if is_unbatched:
        warped = torch.squeeze(warped, dim=0)

    return warped
def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    """Computes affine matrix for translation."""
    matrix = torch.eye(
        3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)

    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix

def translate(tensor: torch.Tensor, translation: torch.Tensor,
              align_corners: bool = False) -> torch.Tensor:

    if not torch.is_tensor(tensor):
        raise TypeError("Input tensor type is not a torch.Tensor. Got {}"
                        .format(type(tensor)))
    if not torch.is_tensor(translation):
        raise TypeError("Input translation type is not a torch.Tensor. Got {}"
                        .format(type(translation)))
    if len(tensor.shape) not in (3, 4,):
        raise ValueError("Invalid tensor shape, we expect CxHxW or BxCxHxW. "
                         "Got: {}".format(tensor.shape))

    # compute the translation matrix
    translation_matrix = _compute_translation_matrix(translation)

    # warp using the affine transform
    return affine(tensor, translation_matrix[..., :2, :3], align_corners=align_corners)




def translation(x,tx, ty):
    # tx, ty \in {-1,0,1}
    ratio = 0.15
    if tx == 1: #translate x
        shift_x = int(x.size(2) * ratio)
    elif tx == -1:
        shift_x = -int(x.size(2) * ratio)
    else: shift_x = 0
    if ty ==1 :
        shift_y = int(x.size(3) * ratio)
    elif ty == -1:
        shift_y = -int(x.size(3) * ratio)
    else:  shift_y = 0
    shift_tensor = torch.ones(x.shape[0],2).cuda()
    shift_tensor[:,0] = shift_x
    shift_tensor[:,1] = shift_y
    # print(shift_tensor)
    # print("shift tensor shape: ", shift_tensor.shape)
    # print("shift tensor: ", shift_tensor)
    x = translate(x,shift_tensor)
    # print("after translation: ", x.shape)
    return x

def h_flip(x, b):
    # b is bool
    if b == True:
        x = F_k.apply_hflip((x))

    return x

def rotate_90(x,k):
    # k is 0,1,2,3
    degreesarr = [0,90, 180, 270, 360]
    degrees = torch.tensor(degreesarr[k]).cuda()
    x = F_k.apply_rotation((x),
        {'degrees': degrees}, {'interpolation': torch.tensor([1]).cuda(), 'align_corners': torch.tensor(True)}).cuda()
    return x


# data = 'cifar'
# scale = 128
# pos_class = 'plane'
# path = str(data) + "_test_scale" + str(scale) + "_" + str(pos_class)
#
# xTest_input = np.load(path + "/" + str(data) + "_data_test_" + str(pos_class) + str(scale) + ".npy")
# real = torch.from_numpy(xTest_input[2]).unsqueeze(0).cuda()
# real = functions.norm(real)
# real = real[:, 0:3, :, :]
# x=real
#
# # count=0
#
# genertator0 = itertools.product((0,),
#                                 (False,),
#                                 (-1,1),
#                                 (-1,),
#                                 (0,))
#
# genertator1 = itertools.product((0,),
#                                 (False, True),
#                                 (0, 1),
#                                 (0, 1),
#                                 (0, 1, 2, 3))
#
# genertator2 = itertools.product((1,),
#                                 (False, True,),
#                                 (0,),
#                                 (0,),
#                                 (0, 1, 2, 3))
#
#
# genertator3 = itertools.product((0,),
#                                 (False,),
#                                 (-1,),
#                                 (1,),
#                                 (0,))
#
# genertator = itertools.chain(genertator0, genertator1,genertator2, genertator3)
# lst = list(genertator)
# random.shuffle(lst)
# print(lst)
# for flag_color,is_flip, tx, ty, k_rotate in lst:
#     count +=1
#     print(flag_color,is_flip, tx, ty, k_rotate)
#     new = apply_transform(x, is_flip, tx, ty, k_rotate,flag_color)
#     plt.imsave("paris_transform" + str(flag_color) + str(is_flip) +"_ " + str(tx) +"_"+ str(ty) + "_" + str(k_rotate)+ ".png",
#                functions.convert_image_np(new.detach())               )
# print(count)