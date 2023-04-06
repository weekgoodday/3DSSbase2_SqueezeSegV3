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
import __init__ as booger # 会查找__init__.py 难怪from tasks.semantic.modules.xxx不报错
import collections
import copy
import cv2
import os
import numpy as np
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.trainer import *
from tasks.semantic.postproc.KNN import KNN
import sys
import pandas
def apply_dropout(m):
  if type(m) == torch.nn.modules.dropout.Dropout2d or type(m) == nn.Dropout():
    m.p=0.2
    m.train()

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,mode="train_split"):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    if(mode=="train_split"):
      using_sequences=self.DATA["split"]["sample"]
    elif(mode=="test_split"):
      using_sequences=self.DATA["split"]["sample2"]
    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=None,
                                      valid_sequences=using_sequences,
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
  def h(self,a): # 直接*就可以 两个立方块对应相乘
    b=torch.zeros(a.shape[1],a.shape[2])
    b=-torch.sum(a*torch.log(a+1e-45),dim=0) #实测torch.log() 一旦小于1e-45就会算出inf 乘0就是 Nan
    return b
  def outputcsv(self,a,str):
    c=a.squeeze(0).numpy()
    df=pandas.DataFrame(c)
    df.to_csv('/home/zht/outputfile/'+str+'.csv')
  def infer_subset(self, loader, to_orig_fn):
    # switch to evaluate mode
    # self.model.eval()
    # zht:开启全部dropout已得到uncertainty
    self.model.eval()
    # for m in self.model.modules():  # 确认是7个0.01的Dropout
    #   if type(m) == torch.nn.modules.dropout.Dropout2d or type(m) == nn.Dropout():
    #     print(m)
    #     print(m.p)
    # import pdb
    # pdb.set_trace()
    self.model.apply(apply_dropout)
    T=10
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
        # zht: 前传T次
        proj_output_T=torch.zeros(T,self.parser.get_n_classes(),proj_in.shape[-2],proj_in.shape[-1]) #T次前传的概率向量 维度(T,20,64,2048)
        proj_total=torch.zeros(T,proj_in.shape[-2],proj_in.shape[-1]) #T次前传的AU
        for i in range(T):
          proj_output, _, _, _, _ = self.model(proj_in, proj_mask)
        # print(proj_output.shape)  # torch.Size([1,20,64,2048])
        # print(proj_output.device)
        # print(proj_output[0].cpu().shape)
          proj_output_T[i,:]=proj_output
          proj_total[i,:]=self.h(proj_output[0].cpu())
        # for i in range(10):
        #   with open('/home/zht/output_file/see10probs.txt','a') as f:
        #     for j in range(20):
        #       f.write('%.2f '% proj_output_T[i,j,0,0])
        #     f.write(f'\n')
        # zht: 计算熵
        
        proj_output_mean=torch.mean(proj_output_T,dim=0) #torch.Size([20, 64, 2048])
        Aleatoric_uncertainty=torch.mean(proj_total,dim=0)
        Total_uncertainty=self.h(proj_output_mean)
        Epistemic_uncertainty=Total_uncertainty-Aleatoric_uncertainty
        # print(pro_labels[0][1,2000:2023])
        # print(Epistemic_uncertainty[1,2000:2023])
        # print(Aleatoric_uncertainty[1,2000:2023])
        # print(Total_uncertainty[1,2000:2023])
        # import pdb 
        # pdb.set_trace()
        # debug：不改成+1e-45验证集会在特定位置报错 
        # print(proj_output_T[:,:,57,993])
        # print(proj_total[:,57,993])
        # print(Aleatoric_uncertainty[57,993])
        # print(Total_uncertainty[57,993])
        
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
        # proj_argmax = proj_output[0].argmax(dim=0)
        proj_argmax = proj_output_mean.argmax(dim=0)
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
        # print("proj_output[0]:",proj_output[0].shape)
        # unproj_output=self.post(proj_range,
        #           unproj_range,
        #           proj_output[0],
        #           p_x,
        #           p_y)
        # print("unproj_output:",unproj_output.shape)
        # if self.post:
        #   # knn postproc  rangenet++的特有KNN后处理算法，能够提高标签的准确性鲁棒性  流程是找最近 投票取最大
        #   # 20分类的概率向量 uncertainty也可以得到修正后的（也是邻居取最大）
        #   # 但暂时没必要 麻烦极了 且没什么物理意义
        #   unproj_argmax = self.post(proj_range,
        #                             unproj_range,
        #                             proj_argmax,
        #                             p_x,
        #                             p_y)
        #   # unproj_uncertainty_bak = self.post(proj_range,
        #   #                                    unproj_range,
        #   #                                    uncertainty_bak,
        #   #                                    p_x,
        #   #                                    p_y
        #   # )
        # else:
        #   # put in original pointcloud using indexes
        #   unproj_argmax = proj_argmax[p_y, p_x]
        # 关键：转变为原始点云
        unproj_AU=Aleatoric_uncertainty[p_y,p_x]
        unproj_EU=Epistemic_uncertainty[p_y,p_x]
        unproj_output=proj_output_mean[:,p_y,p_x]
        unproj_argmax = proj_argmax[p_y,p_x] #zht 强制不开启KNN后处理
        # debug：不改成+1e-45验证集会在特定位置报错 
        # print(unproj_AU[115508])
        # print(unproj_EU[115508])
        # print(p_y[115508])
        # print(p_x[115508])
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
        # print(pred_np) #0-19标签
        # pred_np = to_orig_fn(pred_np) ###zht:注释这句让标签在0-19
        # print(pred_np) #原标签
        # print(pro_labels.shape)
        unproj_label=pro_labels[0][p_y,p_x]
        # print(unproj_label) #这才是groundtruth 0-19版
        # save scan
        path_name1=path_name.split('.')[0]+".pd"
        path1 = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name1)
        pred_np.tofile(path1)

        unproj_label=unproj_label.cpu().numpy().reshape((-1)).astype(np.int32)
        path_name2=path_name.split(".")[0]+".gt"
        path2=os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name2)
        unproj_label.tofile(path2) #0-19的真值

        unproj_output=unproj_output.cpu().numpy().reshape((20,-1)).T.astype(np.float32)
        path_name3=path_name.split(".")[0]+".prob"
        path3=os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name3)
        unproj_output.tofile(path3)

        unproj_AU=unproj_AU.cpu().numpy().reshape((-1)).astype(np.float32)
        path_name4=path_name.split(".")[0]+".au"
        path4=os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name4)
        #raise NotImplementedError
        unproj_AU.tofile(path4)

        unproj_EU=unproj_EU.cpu().numpy().reshape((-1)).astype(np.float32)
        path_name5=path_name.split(".")[0]+".eu"
        path5=os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name5)
        #raise NotImplementedError
        unproj_EU.tofile(path5)
        break
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


