import os
import cv2
import os

from aligned_reid.model.Model import Model
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torch

from aligned_reid.utils.distance import compute_dist, low_memory_matrix_op, parallel_local_dist, normalize, local_dist
from aligned_reid.utils.utils import load_state_dict, measure_time
from aligned_reid.utils.utils import set_devices
from torch.autograd import Variable
import numpy as np


def feature_extraction(model):
    sys_device_ids = (0,)

    TVT, TMO = set_devices(sys_device_ids)

    root_directory = './bounding_boxes/Dynamic_Database/'
    name_list1 = os.listdir(root_directory)
    print name_list1

    global_feature_list = []
    local_feature_list = []
    for name in name_list1:
        glist = []
        llist = []
        directory = './bounding_boxes/Dynamic_Database/' + name + '/'

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

# local_conv_out_channels = 128
# num_classes = 3
#
# model = Model(local_conv_out_channels=local_conv_out_channels, num_classes=num_classes)
#     # Model wrapper
# model_w = DataParallel(model)
#
# base_lr = 2e-4
# weight_decay = 0.0005
# optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
#
# # Bind them together just to save some codes in the following usage.
# modules_optims = [model, optimizer]
#
# model_weight_file = '../../model_weight.pth'
#
# map_location = (lambda storage, loc: storage)
# sd = torch.load(model_weight_file, map_location=map_location)
# load_state_dict(model, sd)
# print('Loaded model weights from {}'.format(model_weight_file))
# g, l, n = feature_extraction(model)
# print g
# print len(g[0])
# print g[0][1]