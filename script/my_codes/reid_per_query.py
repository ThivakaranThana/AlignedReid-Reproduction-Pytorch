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


def find_matching_id (string):
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

    name_list = ['thivakaran/', 'sudeera/', 'hasithap/', 'athavan']
    global_features_list = []
    local_features_list = []
    global_dist = []
    global_local_dist=[]
    local_dist = []
    idlist = []

    root_directory = './bounding_boxes/Dynamic_Database/'
    name_list1= os.listdir(root_directory)
    print name_list1

    for name in name_list:
        glist = []
        llist =[]
        global_features_list = []
        gallery_global_features_list = []
        query_global_features_list = []
        directory = './bounding_boxes/Dynamic_Database/' + name

        for filename in os.listdir(directory):
            for i in range(6):
                if filename.startswith('person_'+str(i)+'.jpg'):
                    input_image = cv2.imread('bounding_boxes/Dynamic_Database/' + name + 'person_' + str(i) + '.jpg')
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
                else:
                    continue
        input_image = cv2.imread('bounding_boxes/Dynamic_Database/query/'+string)
        #input_image = cv2.imread('query/'+string)
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
        glist.append(global_feat)
        # local_features_list.append(local_feat)
        llist.append(local_feat)
        # idlist.append(name)

        # Restore the model to its old train/eval mode.
        model.train(old_train_eval_model)

        ###################
        # Global Distance #
        ###################
        # global_features_list = np.vstack((glist[n-3], glist[n-2],glist[n-1],glist[n]))

        if len(glist) >= 2:
            global_features_list = np.vstack((glist[0], glist[1]))
            local_features_list = np.vstack((llist[0], llist[1]))
            for l in range(len(glist) - 2):
                global_features_list = np.vstack((global_features_list, glist[l + 2]))
                local_features_list = np.vstack((local_features_list, llist[l+2]))

            global_features_list = normalize(global_features_list, axis=1)
            gallery_global_features_list = global_features_list[0:len(glist) - 1]
            query_global_features_list = np.vstack((global_features_list[-1])).T

            local_features_list = normalize(local_features_list, axis=-1)
            gallery_local_features_list = local_features_list[0:len(glist) - 1]
            query_local_features_list = np.expand_dims(local_features_list[-1], axis=0)

            # query-gallery distance using global distance
            global_q_g_dist = compute_dist(query_global_features_list, gallery_global_features_list, type='euclidean')
            #print global_q_g_dist

            # query-gallery distance using local distance
            local_q_g_dist = parallel_local_dist(query_local_features_list, gallery_local_features_list)
            #print local_q_g_dist
            # #

            index_min_g = np.argmin(global_q_g_dist)
            #print index_min_g

            index_min_l = np.argmin(local_q_g_dist)
            #print index_min_l

            global_dist.append(global_q_g_dist[0][index_min_g])
            local_dist.append(local_q_g_dist[0][index_min_l])

            #print global_dist
            #print local_dist

            global_local_distance = global_q_g_dist + local_q_g_dist
            index_min_g_l = np.argmin(global_local_distance)
            global_local_dist.append(global_local_distance[0][index_min_g_l])
            idlist.append(name)

    ans = np.argmin(global_local_dist)
    matchedId = 'query' + string + ' is ' + idlist[ans]
    print 'global', global_dist
    print 'local', local_dist
    print  'global_local', global_local_dist

    return matchedId


Id = find_matching_id('person3.jpg')
print(Id)


