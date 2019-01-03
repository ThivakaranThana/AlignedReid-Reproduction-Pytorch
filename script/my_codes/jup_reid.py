from __future__ import print_function
import cv2

# from aligned_reid.model.Model import Model
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torch

# from aligned_reid.utils.distance import compute_dist, low_memory_matrix_op, parallel_local_dist, normalize, local_dist
# from aligned_reid.utils.utils import load_state_dict, measure_time
# from aligned_reid.utils.utils import set_devices
from torch.autograd import Variable
import numpy as np

#####Model###

import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#####from .resnet import resnet50
import math
import torch.utils.model_zoo as model_zoo
import os
import os.path as osp
import cPickle as pickle
from scipy import io
import datetime
import time
from contextlib import contextmanager
from torch.autograd import Variable
##############################################################main function reid###########
import tensorflow as tf
import copy
import paramiko
import sys,random,datetime
#########################################################

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def remove_fc(state_dict):
    """Remove the fc layer parameters from state_dict."""
    for key, value in state_dict.items():
        if key.startswith('fc.'):
            del state_dict[key]
    return state_dict


def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
    return model


def resnet34(pretrained=False):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
    return model


def resnet50(pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
    return model


class Model(nn.Module):
    def __init__(self, local_conv_out_channels=128, num_classes=None):
        super(Model, self).__init__()
        self.base = resnet50(pretrained=True)
        planes = 2048
        self.local_conv = nn.Conv2d(planes, local_conv_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.local_relu = nn.ReLU(inplace=True)

        if num_classes is not None:
            self.fc = nn.Linear(planes, num_classes)
            init.normal(self.fc.weight, std=0.001)
            init.constant(self.fc.bias, 0)

    def forward(self, x):
        """
        Returns:
          global_feat: shape [N, C]
          local_feat: shape [N, H, c]
        """
        # shape [N, C, H, W]
        feat = self.base(x)
        global_feat = F.avg_pool2d(feat, feat.size()[2:])
        # shape [N, C]
        global_feat = global_feat.view(global_feat.size(0), -1)
        # shape [N, C, H, 1]
        local_feat = torch.mean(feat, -1, keepdim=True)
        local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
        # shape [N, H, c]
        local_feat = local_feat.squeeze(-1).permute(0, 2, 1)

        if hasattr(self, 'fc'):
            logits = self.fc(global_feat)
            return global_feat, local_feat, logits

        return global_feat, local_feat


#####################################


def normalize(nparray, order=2, axis=0):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
      array1: numpy array with shape [m1, n]
      array2: numpy array with shape [m2, n]
      type: one of ['cosine', 'euclidean']
    Returns:
      numpy array with shape [m1, m2]
    """
    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        array1 = normalize(array1, axis=1)
        array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)
        return dist


def shortest_dist(dist_mat):
    """Parallel version.
    Args:
      dist_mat: numpy array, available shape
        1) [m, n]
        2) [m, n, N], N is batch size
        3) [m, n, *], * can be arbitrary additional dimensions
    Returns:
      dist: three cases corresponding to `dist_mat`
        1) scalar
        2) numpy array, with shape [N]
        3) numpy array with shape [*]
    """
    m, n = dist_mat.shape[:2]
    dist = np.zeros_like(dist_mat)
    for i in range(m):
        for j in range(n):
            if (i == 0) and (j == 0):
                dist[i, j] = dist_mat[i, j]
            elif (i == 0) and (j > 0):
                dist[i, j] = dist[i, j - 1] + dist_mat[i, j]
            elif (i > 0) and (j == 0):
                dist[i, j] = dist[i - 1, j] + dist_mat[i, j]
            else:
                dist[i, j] = \
                    np.min(np.stack([dist[i - 1, j], dist[i, j - 1]], axis=0), axis=0) \
                    + dist_mat[i, j]
    # I ran into memory disaster when returning this reference! I still don't
    # know why.
    # dist = dist[-1, -1]
    dist = dist[-1, -1].copy()
    return dist


def meta_local_dist(x, y):
    """
    Args:
      x: numpy array, with shape [m, d]
      y: numpy array, with shape [n, d]
    Returns:
      dist: scalar
    """
    eu_dist = compute_dist(x, y, 'euclidean')
    # numerator = (np.exp(eu_dist) - 1.)
    # denominator = (np.exp(eu_dist) + 1.)
    # dist_mat = numerator / denominator
    dist_mat = (np.exp(eu_dist) - 1.) / (np.exp(eu_dist) + 1.)
    dist = shortest_dist(dist_mat[np.newaxis])[0]
    return dist


# Tooooooo slow!
def serial_local_dist(x, y):
    """
    Args:
      x: numpy array, with shape [M, m, d]
      y: numpy array, with shape [N, n, d]
    Returns:
      dist: numpy array, with shape [M, N]
    """
    M, N = x.shape[0], y.shape[0]
    dist_mat = np.zeros([M, N])
    for i in range(M):
        for j in range(N):
            a = meta_local_dist(x[i], y[j])
            dist_mat[i, j] = a
    return dist_mat


def parallel_local_dist(x, y):
    """Parallel version.
    Args:
      x: numpy array, with shape [M, m, d]
      y: numpy array, with shape [N, n, d]
    Returns:
      dist: numpy array, with shape [M, N]
    """
    M, m, d = x.shape
    N, n, d = y.shape
    x = x.reshape([M * m, d])
    y = y.reshape([N * n, d])
    # shape [M * m, N * n]
    dist_mat = compute_dist(x, y, type='euclidean')
    dist_mat = (np.exp(dist_mat) - 1.) / (np.exp(dist_mat) + 1.)
    # shape [M * m, N * n] -> [M, m, N, n] -> [m, n, M, N]
    dist_mat = dist_mat.reshape([M, m, N, n]).transpose([1, 3, 0, 2])
    # shape [M, N]
    dist_mat = shortest_dist(dist_mat)
    return dist_mat


def local_dist(x, y):
    if (x.ndim == 2) and (y.ndim == 2):
        return meta_local_dist(x, y)
    elif (x.ndim == 3) and (y.ndim == 3):
        return serial_local_dist(x, y)
    else:
        raise NotImplementedError('Input shape not supported.')


def low_memory_matrix_op(
        func,
        x, y,
        x_split_axis, y_split_axis,
        x_num_splits, y_num_splits,
        verbose=False):
    """
    For matrix operation like multiplication, in order not to flood the memory
    with huge data, split matrices into smaller parts (Divide and Conquer).

    Note:
      If still out of memory, increase `*_num_splits`.

    Args:
      func: a matrix function func(x, y) -> z with shape [M, N]
      x: numpy array, the dimension to split has length M
      y: numpy array, the dimension to split has length N
      x_split_axis: The axis to split x into parts
      y_split_axis: The axis to split y into parts
      x_num_splits: number of splits. 1 <= x_num_splits <= M
      y_num_splits: number of splits. 1 <= y_num_splits <= N
      verbose: whether to print the progress

    Returns:
      mat: numpy array, shape [M, N]
    """

    if verbose:
        import sys
        import time
        printed = False
        st = time.time()
        last_time = time.time()

    mat = [[] for _ in range(x_num_splits)]
    for i, part_x in enumerate(
            np.array_split(x, x_num_splits, axis=x_split_axis)):
        for j, part_y in enumerate(
                np.array_split(y, y_num_splits, axis=y_split_axis)):
            part_mat = func(part_x, part_y)
            mat[i].append(part_mat)

            if verbose:
                if not printed:
                    printed = True
                else:
                    # Clean the current line
                    sys.stdout.write("\033[F\033[K")
                print('Matrix part ({}, {}) / ({}, {}), +{:.2f}s, total {:.2f}s'
                      .format(i + 1, j + 1, x_num_splits, y_num_splits,
                              time.time() - last_time, time.time() - st))
                last_time = time.time()
        mat[i] = np.concatenate(mat[i], axis=1)
    mat = np.concatenate(mat, axis=0)
    return mat


#####################################


################################################################

def time_str(fmt=None):
    if fmt is None:
        fmt = '%Y-%m-%d_%H:%M:%S'
    return datetime.datetime.today().strftime(fmt)


def load_pickle(path):
    """Check and load pickle object.
    According to this post: https://stackoverflow.com/a/41733927, cPickle and
    disabling garbage collector helps with loading speed."""
    assert osp.exists(path)
    # gc.disable()
    with open(path, 'rb') as f:
        ret = pickle.load(f)
    # gc.enable()
    return ret


def save_pickle(obj, path):
    """Create dir and save file."""
    may_make_dir(osp.dirname(osp.abspath(path)))
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=2)


def save_mat(ndarray, path):
    """Save a numpy ndarray as .mat file."""
    io.savemat(path, dict(ndarray=ndarray))


def to_scalar(vt):
    """Transform a length-1 pytorch Variable or Tensor to scalar.
    Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]),
    then npx = tx.cpu().numpy() has shape (1,), not 1."""
    if isinstance(vt, Variable):
        return vt.data.cpu().numpy().flatten()[0]
    if torch.is_tensor(vt):
        return vt.cpu().numpy().flatten()[0]
    raise TypeError('Input should be a variable or tensor')


def transfer_optim_state(state, device_id=-1):
    """Transfer an optimizer.state to cpu or specified gpu, which means
    transferring tensors of the optimizer.state to specified device.
    The modification is in place for the state.
    Args:
      state: An torch.optim.Optimizer.state
      device_id: gpu id, or -1 which means transferring to cpu
    """
    for key, val in state.items():
        if isinstance(val, dict):
            transfer_optim_state(val, device_id=device_id)
        elif isinstance(val, Variable):
            raise RuntimeError("Oops, state[{}] is a Variable!".format(key))
        elif isinstance(val, torch.nn.Parameter):
            raise RuntimeError("Oops, state[{}] is a Parameter!".format(key))
        else:
            try:
                if device_id == -1:
                    state[key] = val.cpu()
                else:
                    state[key] = val.cuda(device=device_id)
            except:
                pass


def may_transfer_optims(optims, device_id=-1):
    """Transfer optimizers to cpu or specified gpu, which means transferring
    tensors of the optimizer to specified device. The modification is in place
    for the optimizers.
    Args:
      optims: A list, which members are either torch.nn.optimizer or None.
      device_id: gpu id, or -1 which means transferring to cpu
    """
    for optim in optims:
        if isinstance(optim, torch.optim.Optimizer):
            transfer_optim_state(optim.state, device_id=device_id)


def may_transfer_modules_optims(modules_and_or_optims, device_id=-1):
    """Transfer optimizers/modules to cpu or specified gpu.
    Args:
      modules_and_or_optims: A list, which members are either torch.nn.optimizer
        or torch.nn.Module or None.
      device_id: gpu id, or -1 which means transferring to cpu
    """
    for item in modules_and_or_optims:
        if isinstance(item, torch.optim.Optimizer):
            transfer_optim_state(item.state, device_id=device_id)
        elif isinstance(item, torch.nn.Module):
            if device_id == -1:
                item.cpu()
            else:
                item.cuda(device=device_id)
        elif item is not None:
            print('[Warning] Invalid type {}'.format(item.__class__.__name__))


class TransferVarTensor(object):
    """Return a copy of the input Variable or Tensor on specified device."""

    def __init__(self, device_id=-1):
        self.device_id = device_id

    def __call__(self, var_or_tensor):
        return var_or_tensor.cpu() if self.device_id == -1 \
            else var_or_tensor.cuda(self.device_id)


class TransferModulesOptims(object):
    """Transfer optimizers/modules to cpu or specified gpu."""

    def __init__(self, device_id=-1):
        self.device_id = device_id

    def __call__(self, modules_and_or_optims):
        may_transfer_modules_optims(modules_and_or_optims, self.device_id)


def set_devices(sys_device_ids):
    """
    It sets some GPUs to be visible and returns some wrappers to transferring
    Variables/Tensors and Modules/Optimizers.
    Args:
      sys_device_ids: a tuple; which GPUs to use
        e.g.  sys_device_ids = (), only use cpu
              sys_device_ids = (3,), use the 4th gpu
              sys_device_ids = (0, 1, 2, 3,), use first 4 gpus
              sys_device_ids = (0, 2, 4,), use the 1st, 3rd and 5th gpus
    Returns:
      TVT: a `TransferVarTensor` callable
      TMO: a `TransferModulesOptims` callable
    """
    # Set the CUDA_VISIBLE_DEVICES environment variable
    import os
    visible_devices = ''
    for i in sys_device_ids:
        visible_devices += '{}, '.format(i)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    # Return wrappers.
    # Models and user defined Variables/Tensors would be transferred to the
    # first device.
    #device_id = 0 if len(sys_device_ids) > 0 else -1
    device_id = -1 if len(sys_device_ids) > 0 else 0      #for my lap
    TVT = TransferVarTensor(device_id)
    TMO = TransferModulesOptims(device_id)
    return TVT, TMO


def set_devices_for_ml(sys_device_ids):
    """This version is for mutual learning.

    It sets some GPUs to be visible and returns some wrappers to transferring
    Variables/Tensors and Modules/Optimizers.

    Args:
      sys_device_ids: a tuple of tuples; which devices to use for each model,
        len(sys_device_ids) should be equal to number of models. Examples:

        sys_device_ids = ((-1,), (-1,))
          the two models both on CPU
        sys_device_ids = ((-1,), (2,))
          the 1st model on CPU, the 2nd model on GPU 2
        sys_device_ids = ((3,),)
          the only one model on the 4th gpu
        sys_device_ids = ((0, 1), (2, 3))
          the 1st model on GPU 0 and 1, the 2nd model on GPU 2 and 3
        sys_device_ids = ((0,), (0,))
          the two models both on GPU 0
        sys_device_ids = ((0,), (0,), (1,), (1,))
          the 1st and 2nd model on GPU 0, the 3rd and 4th model on GPU 1

    Returns:
      TVTs: a list of `TransferVarTensor` callables, one for one model.
      TMOs: a list of `TransferModulesOptims` callables, one for one model.
      relative_device_ids: a list of lists; `sys_device_ids` transformed to
        relative ids; to be used in `DataParallel`
    """
    import os

    all_ids = []
    for ids in sys_device_ids:
        all_ids += ids
    unique_sys_device_ids = list(set(all_ids))
    unique_sys_device_ids.sort()
    if -1 in unique_sys_device_ids:
        unique_sys_device_ids.remove(-1)

    # Set the CUDA_VISIBLE_DEVICES environment variable

    visible_devices = ''
    for i in unique_sys_device_ids:
        visible_devices += '{}, '.format(i)
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices

    # Return wrappers

    relative_device_ids = []
    TVTs, TMOs = [], []
    for ids in sys_device_ids:
        relative_ids = []
        for id in ids:
            if id != -1:
                id = find_index(unique_sys_device_ids, id)
            relative_ids.append(id)
        relative_device_ids.append(relative_ids)

        # Models and user defined Variables/Tensors would be transferred to the
        # first device.
        TVTs.append(TransferVarTensor(relative_ids[0]))
        TMOs.append(TransferModulesOptims(relative_ids[0]))
    return TVTs, TMOs, relative_device_ids


def load_ckpt(modules_optims, ckpt_file, load_to_cpu=True, verbose=True):
    """Load state_dict's of modules/optimizers from file.
    Args:
      modules_optims: A list, which members are either torch.nn.optimizer
        or torch.nn.Module.
      ckpt_file: The file path.
      load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers
        to cpu type.
    """
    map_location = (lambda storage, loc: storage) if load_to_cpu else None
    ckpt = torch.load(ckpt_file, map_location=map_location)
    for m, sd in zip(modules_optims, ckpt['state_dicts']):
        m.load_state_dict(sd)
    if verbose:
        print('Resume from ckpt {}, \nepoch {}, \nscores {}'.format(
            ckpt_file, ckpt['ep'], ckpt['scores']))
    return ckpt['ep'], ckpt['scores']


def save_ckpt(modules_optims, ep, scores, ckpt_file):
    """Save state_dict's of modules/optimizers to file.
    Args:
      modules_optims: A list, which members are either torch.nn.optimizer
        or torch.nn.Module.
      ep: the current epoch number
      scores: the performance of current model
      ckpt_file: The file path.
    Note:
      torch.save() reserves device type and id of tensors to save, so when
      loading ckpt, you have to inform torch.load() to load these tensors to
      cpu or your desired gpu, if you change devices.
    """
    state_dicts = [m.state_dict() for m in modules_optims]
    ckpt = dict(state_dicts=state_dicts,
                ep=ep,
                scores=scores)
    may_make_dir(osp.dirname(osp.abspath(ckpt_file)))
    torch.save(ckpt, ckpt_file)


def load_state_dict(model, src_state_dict):
    """Copy parameters and buffers from `src_state_dict` into `model` and its
    descendants. The `src_state_dict.keys()` NEED NOT exactly match
    `model.state_dict().keys()`. For dict key mismatch, just
    skip it; for copying error, just output warnings and proceed.

    Arguments:
      model: A torch.nn.Module object.
      src_state_dict (dict): A dict containing parameters and persistent buffers.
    Note:
      This is modified from torch.nn.modules.module.load_state_dict(), to make
      the warnings and errors more detailed.
    """
    from torch.nn import Parameter

    dest_state_dict = model.state_dict()
    for name, param in src_state_dict.items():
        if name not in dest_state_dict:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            dest_state_dict[name].copy_(param)
        except Exception, msg:
            print("Warning: Error occurs when copying '{}': {}"
                  .format(name, str(msg)))

    src_missing = set(dest_state_dict.keys()) - set(src_state_dict.keys())
    if len(src_missing) > 0:
        print("Keys not found in source state_dict: ")
        for n in src_missing:
            print('\t', n)

    dest_missing = set(src_state_dict.keys()) - set(dest_state_dict.keys())
    if len(dest_missing) > 0:
        print("Keys not found in destination state_dict: ")
        for n in dest_missing:
            print('\t', n)


def is_iterable(obj):
    return hasattr(obj, '__len__')


def may_set_mode(maybe_modules, mode):
    """maybe_modules: an object or a list of objects."""
    assert mode in ['train', 'eval']
    if not is_iterable(maybe_modules):
        maybe_modules = [maybe_modules]
    for m in maybe_modules:
        if isinstance(m, torch.nn.Module):
            if mode == 'train':
                m.train()
            else:
                m.eval()


def may_make_dir(path):
    """
    Args:
      path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
    Note:
      `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
    """
    # This clause has mistakes:
    # if path is None or '':

    if path in [None, '']:
        return
    if not osp.exists(path):
        os.makedirs(path)


class AverageMeter(object):
    """Modified from Tong Xiao's open-reid.
    Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / (self.count + 1e-20)


class RunningAverageMeter(object):
    """Computes and stores the running average and current value"""

    def __init__(self, hist=0.99):
        self.val = None
        self.avg = None
        self.hist = hist

    def reset(self):
        self.val = None
        self.avg = None

    def update(self, val):
        if self.avg is None:
            self.avg = val
        else:
            self.avg = self.avg * self.hist + val * (1 - self.hist)
        self.val = val


class RecentAverageMeter(object):
    """Stores and computes the average of recent values."""

    def __init__(self, hist_size=100):
        self.hist_size = hist_size
        self.fifo = []
        self.val = 0

    def reset(self):
        self.fifo = []
        self.val = 0

    def update(self, val):
        self.val = val
        self.fifo.append(val)
        if len(self.fifo) > self.hist_size:
            del self.fifo[0]

    @property
    def avg(self):
        assert len(self.fifo) > 0
        return float(sum(self.fifo)) / len(self.fifo)


def get_model_wrapper(model, multi_gpu):
    from torch.nn.parallel import DataParallel
    if multi_gpu:
        return DataParallel(model)
    else:
        return model


class ReDirectSTD(object):
    """Modified from Tong Xiao's `Logger` in open-reid.
    This class overwrites sys.stdout or sys.stderr, so that console logs can
    also be written to file.
    Args:
      fpath: file path
      console: one of ['stdout', 'stderr']
      immediately_visible: If `False`, the file is opened only once and closed
        after exiting. In this case, the message written to file may not be
        immediately visible (Because the file handle is occupied by the
        program?). If `True`, each writing operation of the console will
        open, write to, and close the file. If your program has tons of writing
        operations, the cost of opening and closing file may be obvious. (?)
    Usage example:
      `ReDirectSTD('stdout.txt', 'stdout', False)`
      `ReDirectSTD('stderr.txt', 'stderr', False)`
    NOTE: File will be deleted if already existing. Log dir and file is created
      lazily -- if no message is written, the dir and file will not be created.
    """

    def __init__(self, fpath=None, console='stdout', immediately_visible=False):
        import sys
        import os
        import os.path as osp

        assert console in ['stdout', 'stderr']
        self.console = sys.stdout if console == 'stdout' else sys.stderr
        self.file = fpath
        self.f = None
        self.immediately_visible = immediately_visible
        if fpath is not None:
            # Remove existing log file.
            if osp.exists(fpath):
                os.remove(fpath)

        # Overwrite
        if console == 'stdout':
            sys.stdout = self
        else:
            sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            may_make_dir(os.path.dirname(osp.abspath(self.file)))
            if self.immediately_visible:
                with open(self.file, 'a') as f:
                    f.write(msg)
            else:
                if self.f is None:
                    self.f = open(self.file, 'w')
                self.f.write(msg)

    def flush(self):
        self.console.flush()
        if self.f is not None:
            self.f.flush()
            import os
            os.fsync(self.f.fileno())

    def close(self):
        self.console.close()
        if self.f is not None:
            self.f.close()


def set_seed(seed):
    import random
    random.seed(seed)
    print('setting random-seed to {}'.format(seed))

    import numpy as np
    np.random.seed(seed)
    print('setting np-random-seed to {}'.format(seed))

    import torch
    torch.backends.cudnn.enabled = False
    print('cudnn.enabled set to {}'.format(torch.backends.cudnn.enabled))
    # set seed for CPU
    torch.manual_seed(seed)
    print('setting torch-seed to {}'.format(seed))


def print_array(array, fmt='{:.2f}', end=' '):
    """Print a 1-D tuple, list, or numpy array containing digits."""
    s = ''
    for x in array:
        s += fmt.format(float(x)) + end
    s += '\n'
    print(s)
    return s


# Great idea from https://github.com/amdegroot/ssd.pytorch
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def tight_float_str(x, fmt='{:.4f}'):
    return fmt.format(x).rstrip('0').rstrip('.')


def find_index(seq, item):
    for i, x in enumerate(seq):
        if item == x:
            return i
    return -1


def adjust_lr_exp(optimizer, base_lr, ep, total_ep, start_decay_at_ep):
    """Decay exponentially in the later phase of training. All parameters in the
    optimizer share the same learning rate.

    Args:
      optimizer: a pytorch `Optimizer` object
      base_lr: starting learning rate
      ep: current epoch, ep >= 1
      total_ep: total number of epochs to train
      start_decay_at_ep: start decaying at the BEGINNING of this epoch

    Example:
      base_lr = 2e-4
      total_ep = 300
      start_decay_at_ep = 201
      It means the learning rate starts at 2e-4 and begins decaying after 200
      epochs. And training stops after 300 epochs.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep < start_decay_at_ep:
        return

    for g in optimizer.param_groups:
        g['lr'] = (base_lr * (0.001 ** (float(ep + 1 - start_decay_at_ep)
                                        / (total_ep + 1 - start_decay_at_ep))))
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


def adjust_lr_staircase(optimizer, base_lr, ep, decay_at_epochs, factor):
    """Multiplied by a factor at the BEGINNING of specified epochs. All
    parameters in the optimizer share the same learning rate.

    Args:
      optimizer: a pytorch `Optimizer` object
      base_lr: starting learning rate
      ep: current epoch, ep >= 1
      decay_at_epochs: a list or tuple; learning rate is multiplied by a factor
        at the BEGINNING of these epochs
      factor: a number in range (0, 1)

    Example:
      base_lr = 1e-3
      decay_at_epochs = [51, 101]
      factor = 0.1
      It means the learning rate starts at 1e-3 and is multiplied by 0.1 at the
      BEGINNING of the 51'st epoch, and then further multiplied by 0.1 at the
      BEGINNING of the 101'st epoch, then stays unchanged till the end of
      training.

    NOTE:
      It is meant to be called at the BEGINNING of an epoch.
    """
    assert ep >= 1, "Current epoch number should be >= 1"

    if ep not in decay_at_epochs:
        return

    ind = find_index(decay_at_epochs, ep)
    for g in optimizer.param_groups:
        g['lr'] = base_lr * factor ** (ind + 1)
    print('=====> lr adjusted to {:.10f}'.format(g['lr']).rstrip('0'))


@contextmanager
def measure_time(enter_msg):
    st = time.time()
    print(enter_msg)
    yield
    print('Done, {:.2f}s'.format(time.time() - st))


################################################################
def feature_extraction(model):
    sys_device_ids = (0,)

    TVT, TMO = set_devices(sys_device_ids)

    root_directory = '/home/madhushanb/sphereface/Dynamic_Database/'
    name_list1 = os.listdir(root_directory)
    print (name_list1)

    global_feature_list = []
    local_feature_list = []
    for name in name_list1:
        glist = []
        llist = []
        directory = '/home/madhushanb/sphereface/Dynamic_Database/' + name + '/'

        for filename in os.listdir(directory):
            input_image = cv2.imread(directory+filename)
            resized_image = cv2.resize(input_image, (128, 256))

            transposed = resized_image.transpose(2, 0, 1)
            test_img = transposed[np.newaxis]

            old_train_eval_model = model.training

            # Set eval mode.
            # Force all BN layers to use global mean and variance, also disable dropout.
            model.eval()

            # ims = np.stack(input_image, axis=0)

            ims = Variable(TVT(torch.from_numpy(test_img).float()))
            global_feat, local_feat = model(ims)[:2]
            global_feat = global_feat.data.cpu().numpy()[0]
            local_feat = local_feat.data.cpu().numpy()

            # global_features_list.append(global_feat)
            glist.append(global_feat)
            # local_features_list.append(local_feat)
            llist.append(local_feat)
            # idlist.append(name)

            # Restore the model to its old train/eval mode.
            model.train(old_train_eval_model)

        global_feature_list.append(glist)
        local_feature_list.append(llist)
    return global_feature_list, local_feature_list, name_list1
################################################################################################
def find_matching_id (global_features_list, local_features_list, name_list, string, model):

    sys_device_ids = (0,)

    TVT, TMO = set_devices(sys_device_ids)

    global_local_dist = []

    for i in range(len(global_features_list)):
        input_image = cv2.imread('bounding_boxes/query/' + string)
        # input_image = cv2.imread('query/'+string)
        resized_image = cv2.resize(input_image, (128, 256))

        transposed = resized_image.transpose(2, 0, 1)
        test_img = transposed[np.newaxis]

        old_train_eval_model = model.training
        model.eval()
        ims = Variable(TVT(torch.from_numpy(test_img).float()))
        global_feat, local_feat = model(ims)[:2]
        global_feat = global_feat.data.cpu().numpy()[0]
        local_feat = local_feat.data.cpu().numpy()

        # global_features_list.append(global_feat)
        global_features_list[i].append(global_feat)
        local_features_list[i].append(local_feat)
        # Restore the model to its old train/eval mode.
        model.train(old_train_eval_model)

        ###################
        # Global Distance #
        ###################

        if len(global_features_list[i]) >= 2:
            global_list = np.vstack((global_features_list[i][0], global_features_list[i][1]))
            local_list = np.vstack((local_features_list[i][0], local_features_list[i][1]))
            for l in range(len(global_features_list[i]) - 2):
                global_list = np.vstack((global_list, global_features_list[i][l + 2]))
                local_list = np.vstack((local_list, local_features_list[i][l + 2]))

            global_list = normalize(global_list, axis=1)
            gallery_global_features_list = global_list[0:len(global_features_list[i]) - 1]
            query_global_features_list = np.vstack((global_list[-1])).T

            local_list = normalize(local_list, axis=-1)
            gallery_local_features_list = local_list[0:len(local_features_list[i]) - 1]
            query_local_features_list = np.expand_dims(local_list[-1], axis=0)

            # query-gallery distance using global distance
            global_q_g_dist = compute_dist(query_global_features_list, gallery_global_features_list, type='euclidean')
            #print 'global ', global_q_g_dist

            # query-gallery distance using local distance
            local_q_g_dist = parallel_local_dist(query_local_features_list, gallery_local_features_list)
            #print 'local ',  local_q_g_dist
            global_local_distance = global_q_g_dist + local_q_g_dist
            #print 'total', global_local_distance
            index_min_g_l = np.argmin(global_local_distance)
            global_local_dist.append(global_local_distance[0][index_min_g_l])

    #print "global local dis", global_local_dist
    ans = np.argmin(global_local_dist)
    matchedId = 'query' + string + ' is ' + name_list[ans]
    return matchedId


###############################person dtector and nms##################
class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time - start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1] * im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3] * im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()


def nms(boxes, overlap_threshold=0.2, mode='union'):
    """Non-maximum suppression.

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.

    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter / np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter / (area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick
#################################################################

######################mainfunction#######################
if __name__ == '__main__':

    local_conv_out_channels = 128
    num_classes = 3
    model = Model(local_conv_out_channels=local_conv_out_channels, num_classes=num_classes)
    # Model wrapper
    model_w = DataParallel(model)

    base_lr = 2e-4
    weight_decay = 0.0005
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Bind them together just to save some codes in the following usage.
    modules_optims = [model, optimizer]

    model_weight_file = '../../model_weight.pth'

    map_location = (lambda storage, loc: storage)
    sd = torch.load(model_weight_file, map_location=map_location)
    load_state_dict(model, sd)
    print('Loaded model weights from {}'.format(model_weight_file))
##################################################################################################################
    model_path = '../../faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    capture = cv2.VideoCapture('custom_video/test5.mp4')
    frameCounter = 0
    f = open("result", "w")
    global_features_list, local_features_list, name_list = feature_extraction(model)
    try:
        while True:
            # Retrieve the latest image from the webcam
            rc, fullSizeBaseImage = capture.read()
            # print height, width
             # Resize the image to 320x240
            (h,w) = fullSizeBaseImage.shape[:2]
            M=cv2.getRotationMatrix2D((w/2,h/2),-90,1)
            baseImage=cv2.warpAffine(fullSizeBaseImage,M,(h,w))
            #baseImage = cv2.resize(fullSizeBaseImage, (1280, 720))
            #baseImage=fullSizeBaseImage

            # Check if a key was pressed and if it was Q, then break
            # from the infinite loop
            #pressedKey = cv2.waitKey(2)
            #if pressedKey == ord('Q'):
            #    break
            frameCounter += 1
            # Every 10 frames, we will have to determine which faces
            # are present in the frame
            if (frameCounter % 10) == 0:
                boxes, scores, classes, num = odapi.processFrame(baseImage)

                nms_input = np.empty((len(boxes), 5))

                # print boxes[:, 1]

                # nms_input[:, :-1] = boxes
                nms_input[:, 0] = [row[1] for row in boxes]
                nms_input[:, 1] = [row[0] for row in boxes]
                nms_input[:, 2] = [row[3] for row in boxes]
                nms_input[:, 3] = [row[2] for row in boxes]
                nms_input[:, 4] = scores

                picks_from_nms = nms(nms_input)

                for i in picks_from_nms:
                    global_features_list_copy = []
                    local_features_list_copy = []
                    if classes[i] == 1 and scores[i] > threshold:
                        box = boxes[i]
                        person_bounding_box = baseImage[box[0]:box[2], box[1]:box[3]]
                        image = "person" + str(i)+'_frameNo_'+str(frameCounter) + '.jpg'
                        cv2.imwrite("bounding_boxes/query/person" + str(i)+'_frameNo_'+str(frameCounter) +'.jpg', person_bounding_box)
                        global_features_list_copy = copy.deepcopy(global_features_list)
                        local_features_list_copy = copy.deepcopy(local_features_list)
                        Id = find_matching_id(global_features_list_copy,
                                              local_features_list_copy, name_list, str(image), model)
                        f.write(Id+"\n")
    except TypeError:
        print ('finished processing')
    except AttributeError:
        print('finished processing')

exit(0)