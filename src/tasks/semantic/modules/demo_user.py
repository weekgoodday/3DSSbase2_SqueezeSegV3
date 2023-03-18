#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.trainer import *
from tasks.semantic.postproc.KNN import KNN

import pandas
class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=None,
                                      valid_sequences=self.DATA["split"]["sample"],
                                      test_sequences=None,
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
      self.model = Segmentator(self.ARCH,
                               self.parser.get_n_classes(),
                               self.modeldir)

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())
    # print(self.ARCH["post"]["KNN"]["params"])
    # print(self.parser.get_n_classes())
    # raise NotImplementedError
    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    # do test set
    self.infer_subset(loader=self.parser.get_valid_set(),
                      to_orig_fn=self.parser.to_original)

    print('Finished Infering')

    return
  def h(self,a):
    # print(a)
    # print(type(a))
    b=torch.zeros(1,a.shape[1],a.shape[2])
    for i in range(a.shape[1]):
      for j in range(a.shape[2]):
        b[0,i,j]=-torch.sum(a[:,i,j]*torch.log(a[:,i,j]))
    return b
  def outputcsv(self,a,str):
    c=a.squeeze(0).numpy()
    df=pandas.DataFrame(c)
    df.to_csv('/home/zht/outputfile/'+str+'.csv')
  def infer_subset(self, loader, to_orig_fn):
    # switch to evaluate mode
    self.model.eval()

    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, pro_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          proj_mask = proj_mask.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()

          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        proj_output, _, _, _, _ = self.model(proj_in, proj_mask)
        # print(proj_output.shape)  # torch.Size([1,20,64,2048])
        # print(proj_output.device)
        # print(proj_output[0].cpu().shape)

        # zht: 计算朴素的熵
        uncertainty_bak=self.h(proj_output[0].cpu())

        # c=uncertainty_bak.squeeze(0).numpy()
        # df=pandas.DataFrame(c)
        #df.to_csv('/home/zht/outputfile/uncertain.csv')
        # with open('/home/zht/outputfile/uncertain.txt', 'a+') as f:
        #   torch.set_printoptions(profile="full")
        #   print(b, file=f)

        # print(b)
        # print(b.shape)

        #raise NotImplementedError
        # 原本就有，利用(1,20,64,2048)的segmentation结果，argmax得到预测结果 gpu上 (64,2048)
        proj_argmax = proj_output[0].argmax(dim=0)

        # print(proj_argmax.device)
        # print(pro_labels.device)
        #matrix_bool=(proj_argmax==pro_labels)
        # self.outputcsv(pro_labels,'label')  暂时不需要输出到csv
        # self.outputcsv(proj_mask.cpu(),'mask')
        # self.outputcsv(proj_argmax.cpu(),'outcn')
        # print(pro_labels)
        # print(proj_mask)
        # print(pro_labels.shape)
        # print(proj_mask.shape)
        # a=torch.ones_like(pro_labels)
        # print(a[matrix_bool].shape)
        # print(proj_argmax.shape)
        # print(uncertainty_bak.shape)
        uncertainty_bak=uncertainty_bak.squeeze(0).cuda()
        #raise NotImplementedError
        # print("proj_output[0]:",proj_output[0].shape)
        # unproj_output=self.post(proj_range,
        #           unproj_range,
        #           proj_output[0],
        #           p_x,
        #           p_y)
        # print("unproj_output:",unproj_output.shape)
        if self.post:
          # knn postproc  rangenet++的特有KNN后处理算法，能够提高标签的准确性鲁棒性  流程是找最近 投票取最大
          # 20分类的概率向量 uncertainty也可以得到修正后的（也是邻居取最大）
          # 但暂时没必要 麻烦极了 且没什么物理意义
          unproj_argmax = self.post(proj_range,
                                    unproj_range,
                                    proj_argmax,
                                    p_x,
                                    p_y)
          # unproj_uncertainty_bak = self.post(proj_range,
          #                                    unproj_range,
          #                                    uncertainty_bak,
          #                                    p_x,
          #                                    p_y
          # )
        else:
          # put in original pointcloud using indexes
          unproj_argmax = proj_argmax[p_y, p_x]
        # 关键：转变为原始点云
        unproj_uncertainty_bak=uncertainty_bak[p_y,p_x]
        unproj_output=proj_output[0][:,p_y,p_x]
        # print("original output prob:", unproj_output.shape) [20,124668]
        if torch.cuda.is_available():
         torch.cuda.synchronize()

        print("Infered seq", path_seq, "scan", path_name,
              "in", time.time() - end, "sec")
        
        end = time.time()

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)
        # print(pred_np.shape) 已经是长度为点数的一维向量
        # print(pred_np)
        # map to original label
        pred_np = to_orig_fn(pred_np)
        # save scan
        path = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name)
        pred_np.tofile(path)
        unproj_uncertainty_bak=unproj_uncertainty_bak.cpu().numpy().reshape((-1)).astype(np.float32)
        path_name2=path_name.split(".")[0]+".uncertainty"
        path2=os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name2)
        #raise NotImplementedError
        unproj_uncertainty_bak.tofile(path2)
        print(unproj_output.shape)
        print(unproj_output[:,0])
        unproj_output=unproj_output.cpu().numpy().reshape((20,-1)).astype(np.float32)
        path_name3=path_name.split(".")[0]+".prob"
        path3=os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name3)
        unproj_output.tofile(path3)
        # raise NotImplementedError
        # 以下画投影视角的图不重要了
        # depth = (cv2.normalize(proj_in[0][0].cpu().numpy(), None, alpha=0, beta=1,
        #                    norm_type=cv2.NORM_MINMAX,
        #                    dtype=cv2.CV_32F) * 255.0).astype(np.uint8)
        # print(depth.shape, proj_mask.shape,proj_argmax.shape)
        # out_img = cv2.applyColorMap(
        #     depth, Trainer.get_mpl_colormap('viridis')) * proj_mask[0].cpu().numpy()[..., None]
        #  # make label prediction
        # pred_color = self.parser.to_color((proj_argmax.cpu().numpy() * proj_mask[0].cpu().numpy()).astype(np.int32))
        # out_img = np.concatenate([out_img, pred_color], axis=0)
        # print(path)
        # cv2.imwrite(path[:-6]+'.png',out_img)


