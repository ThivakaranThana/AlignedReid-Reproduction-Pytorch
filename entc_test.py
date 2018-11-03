from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

from dataset import ImageDataset
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
# import matplotlib.pyplot as plt

from src import detect_faces, show_bboxes
from PIL import Image

#function for aligning faces------------------------------------------------------

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

#fn for loading Database-----------------------------------------------------------

def LoadDatabase(path ='./img_db'):
    dbImages = {}
    
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            s = path+'/'+filename
            dbImages[filename] = cv2.imread(s)
    return dbImages

# load DB landmarks-------------------------------------------------------------------

def LoadDbLandmarks(txtfile = './entcDbLandmarks.txt'):
    dbLandmarkFile = open(txtfile,'r')
    dbLandmarks = {}
    for line in dbLandmarkFile:
        line = line[:-1]
        temp = line.split('\t')
        name = temp.pop(0)
        dbLandmarks[name] = np.array(temp)
    dbLandmarkFile.close()
    return dbLandmarks

#align DB images---------------------------------------------------------------------

def AlignDbImages(dbImages,dbLandmarks):
    dbAlignedImages = {}
    for key in dbImages:
        dbAlignedImages[key] = alignment(dbImages[key],dbLandmarks[key])
    return dbAlignedImages

#load test image--------------------------------------------------------------------

def LoadTestImage(path):
    pilImg = Image.open(path)
    cv2Img = cv2.imread(path)
    return pilImg, cv2Img

#detect landmarks------------------------------------------------------------------

def DetectLandmarks(pilImg):
    testLandmarkList = {}
    boundingBoxesList = []

    bounding_boxes, testLandmarks = detect_faces(pilImg,min_face_size=10.0)
    for face in range(len(testLandmarks)):
        i = testLandmarks[face]
        landmarkReshaped = [i[0],i[5],i[1],i[6],i[2],i[7],i[3],i[8],i[4],i[9]]
        landmarkReshaped = np.around(landmarkReshaped).astype(int)
        a= 'face_'+str(face)
        testLandmarkList[a] = landmarkReshaped
    boundingBoxesList = bounding_boxes
        
    return testLandmarkList, boundingBoxesList

#align test faces-------------------------------------------------------------------

def AlignTestFaces(cv2img, testLandmarkList):
    alignedImages = {}
    for key in testLandmarkList:
        alignedImages[key] = alignment(cv2img,testLandmarkList[key])
    return alignedImages

#recognize face---------------------------------------------------------------------

def RecognizeFace(testAlignedFacesList, dbAlignedFacesList, net):
    nameList={}
    for key1 in testAlignedFacesList:
        testFace = testAlignedFacesList[key1]
        predicts = []
        for key2 in dbAlignedFacesList:
            imglist = [testFace,cv2.flip(testFace,1),dbAlignedFacesList[key2],cv2.flip(dbAlignedFacesList[key2],1)]
            
            for i in range(len(imglist)):
                imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1,3,112,96))
                imglist[i] = (imglist[i]-127.5)/128.0
            
            img = np.vstack(imglist)
            img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
            output = net(img)        
            f = output.data
            f1,f2 = f[0],f[2]
            cosdistance = f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)

            #predicts.append('{}\t{}\t{}\n'.format('niru',key,cosdistance))   
            predicts.append([key2,cosdistance.data.tolist()]) 

        sortedList = sorted(predicts,key=lambda l:l[1], reverse=True)
        
        index = int(key1.split('_')[-1])
        if(sortedList[1][1]>0.6):
            nameList[index] = "multiple Detects"+sortedList[0][0]+ " " +sortedList[1][0]
            
        elif(sortedList[0][1]<0.5):
            nameList[index] = "unknown Best Guess: " + sortedList[0][0]
            
        else:
            nameList[index] = sortedList[0][0]
            
    return nameList

#main-----------------------------------------------------------------------------

def load_system(model_path, database_path, landmarkTextFile):
    model = model_path # 'model/sphere20a_20171020.pth'
    net = 'sphere20a'
    net = getattr(net_sphere, net)()
    net.load_state_dict(torch.load(model))
    net.cuda()
    net.eval()
    net.feature = True

    Database = LoadDatabase(database_path)
    DBLandmarks = LoadDbLandmarks(landmarkTextFile)
    AlignedDB = AlignDbImages(Database, DBLandmarks)
    return net, AlignedDB

def run_test(testImageCV2, net, AlignedDB):
    # load sphereface model-----------------------------------------------------------=



    # pilImage, cv2Image = LoadTestImage(testImagePath) #'./test_photos/7.jpg'
    pilImage = Image.fromarray(testImageCV2)
    testLandMarkList, testBBList = DetectLandmarks(pilImage)
    AlignedTestFaces = AlignTestFaces(testImageCV2,testLandMarkList)

    nameList = RecognizeFace(AlignedTestFaces, AlignedDB, net)

    # show_bboxes(bounding_boxes=testBBList,img=pilImage,names=nameList.values())

    return nameList



