import cv2
import os
import copy

from aligned_reid.model.Model import Model
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torch

from aligned_reid.utils.distance import compute_dist, low_memory_matrix_op, parallel_local_dist, normalize, local_dist
from aligned_reid.utils.utils import load_state_dict, measure_time
from aligned_reid.utils.utils import set_devices
from torch.autograd import Variable
import numpy as np
from feature_extraction import feature_extraction


def find_matching_id (global_features_list, local_features_list, name_list, string, model):
    # local_conv_out_channels = 128
    # num_classes = 3
    #
    # model = Model(local_conv_out_channels=local_conv_out_channels, num_classes=num_classes)
    # # Model wrapper
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


# local_conv_out_channels = 128
# num_classes = 3
#
# model = Model(local_conv_out_channels=local_conv_out_channels, num_classes=num_classes)
# # Model wrapper
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
#
# g, l, n = feature_extraction(model)
# # for i in range(1, 5, 1):
# #     gc = g[:]
# #     lc = l[:]
# #     id = find_matching_id(gc, lc, n, 'person' + str(i)+'.jpg')
# #     print id
#
# print g[0],
# import copy
# gc= copy.deepcopy(g)
# lc= copy.deepcopy(l)
# id = find_matching_id(gc, lc, n, 'athavan4.jpg',model)
# print id
# print 'new', g[0]
#
# # gc = g[:]
# # lc = l[:]
# # id1 = find_matching_id(gc, lc, n, 'hasitha1.jpg',model)
# # print id1
#


