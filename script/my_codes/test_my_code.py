import cv2

from aligned_reid.model.Model import Model
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torch

from aligned_reid.utils.distance import compute_dist, low_memory_matrix_op, local_dist, normalize
from aligned_reid.utils.utils import load_state_dict, measure_time
from aligned_reid.utils.utils import set_devices
from torch.autograd import Variable
import numpy as np



###########
# Models  #
###########
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

sys_device_ids = (0,)

TVT, TMO = set_devices(sys_device_ids)

name_list = ['gow.jpg', 'gunes.jpg', 'niru.jpg', 'gow_query.jpg']

global_features_list = []
local_features_list = []

for name in name_list:

    input_image = cv2.imread('my_images/' + name)
    resized_image = cv2.resize(input_image, (128, 256))

    # input_image = np.asarray(Image.open('gow_query.jpg'))
    transposed = resized_image.transpose(2,0,1)
    test_img = transposed[np.newaxis]

    # cv2.imshow("preview", input_image)
    # key = cv2.waitKey(0)

    old_train_eval_model = model.training

    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable dropout.
    model.eval()

    ims = np.stack(input_image, axis=0)

    ims = Variable(TVT(torch.from_numpy(test_img).float()))
    global_feat, local_feat = model(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()[0]
    local_feat = local_feat.data.cpu().numpy()

    global_features_list.append(global_feat)
    local_features_list.append(local_feat)

    # Restore the model to its old train/eval mode.
    model.train(old_train_eval_model)


###################
# Global Distance #
###################

global_features_list = np.vstack((global_features_list[0], global_features_list[1], global_features_list[2], global_features_list[3]))
global_features_list = normalize(global_features_list, axis=1)

gallery_global_features_list = global_features_list[0:3]
query_global_features_list = np.vstack((global_features_list[3])).T

# query-gallery distance using global distance
global_q_g_dist = compute_dist(query_global_features_list, gallery_global_features_list , type='euclidean')


print global_q_g_dist
#
# ##################
# # Local Distance #
# ##################
#
# local_features_list = np.vstack((local_features_list[0], local_features_list[1], local_features_list[2], local_features_list[3]))
# normalized_local_features_list = normalize(local_features_list, axis=-1)
#
# gallery_local_features_list =normalized_local_features_list[0:3]
# query_local_features_list = np.expand_dims(normalized_local_features_list[3], axis=0)
#
# # A helper function just for avoiding code duplication.
# def low_memory_local_dist(x, y):
#     with measure_time('Computing local distance...'):
#         x_num_splits = int(len(x) / 200) + 1
#         y_num_splits = int(len(y) / 200) + 1
#         z = low_memory_matrix_op(
#             local_dist, x, y, 0, 0, x_num_splits, y_num_splits, verbose=True)
#     return z
#
# # query-gallery distance using local distance
# local_q_g_dist = low_memory_local_dist(query_local_features_list, gallery_local_features_list)
#
# print local_q_g_dist
#
# global_local_distance = global_q_g_dist + local_q_g_dist
#
# print global_local_distance