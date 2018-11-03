import cv2

from aligned_reid.model.Model import Model
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torch
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import set_devices
from torch.autograd import Variable
import numpy as np

input_image = cv2.imread('gow_query.jpg')
resized_image = cv2.resize(input_image, (128, 256))


# input_image = np.asarray(Image.open('gow_query.jpg'))
transposed = resized_image.transpose(2,0,1)
test_img = transposed[np.newaxis]

# cv2.imshow("preview", input_image)
# key = cv2.waitKey(0)

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
optimizer = optim.Adam(model.parameters(), lr = base_lr, weight_decay = weight_decay)

# Bind them together just to save some codes in the following usage.
modules_optims = [model, optimizer]

model_weight_file = '/home/niruhan/AlignedReID-Re-Production-Pytorch/model_weight.pth'

map_location = (lambda storage, loc: storage)
sd = torch.load(model_weight_file, map_location=map_location)
load_state_dict(model, sd)
print('Loaded model weights from {}'.format(model_weight_file))

sys_device_ids = (0,)

TVT, TMO = set_devices(sys_device_ids)

old_train_eval_model = model.training

# Set eval mode.
# Force all BN layers to use global mean and variance, also disable dropout.
model.eval()

ims = np.stack(input_image, axis=0)

ims = Variable(TVT(torch.from_numpy(test_img).float()))
global_feat, local_feat = model(ims)[:2]
global_feat = global_feat.data.cpu().numpy()
local_feat = local_feat.data.cpu().numpy()

# Restore the model to its old train/eval mode.
model.train(old_train_eval_model)




