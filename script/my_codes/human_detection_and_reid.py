import numpy as np
import tensorflow as tf
import cv2
import time

from aligned_reid.model.Model import Model
from torch.nn.parallel import DataParallel
import torch.optim as optim
import torch
from aligned_reid.utils.utils import load_state_dict
from aligned_reid.utils.utils import set_devices
from torch.autograd import Variable

# human detection part

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

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = '../../faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    # cap = cv2.VideoCapture('/path/to/input/video')

    img = cv2.imread('my_images/sutha.jpg')

    # img = cv2.resize(cap, (1280, 720))

    boxes, scores, classes, num = odapi.processFrame(img)

    # Visualization of the results of a detection.

    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            print box[1]
            person_image = img[box[0]:box[2], box[1]:box[3]]

    # cv2.imshow("preview", img)
    # key = cv2.waitKey(0)
    # cv2.imshow("preview", person_image)
    # key = cv2.waitKey(0)



# extracting global features for person reid

input_image = person_image
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

model_weight_file = '../../model_weight.pth'

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


