#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
import time
import numpy as np
import sys
sys.path.append("..")
sys.path.append("../..") 
from tasks.semantic.modules.segmentator import *
from tasks.semantic.modules.trainer import *
import pandas
import argparse
from tasks.semantic.dataset.kitti.parser import Parser
class Demo():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,mode="valid_split"):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    if(mode=="train_split"):
      using_sequences=self.DATA["split"]["train"]
    elif(mode=="test_split"):
      using_sequences=self.DATA["split"]["test"]
    elif(mode=="valid_split"):
      using_sequences=self.DATA["split"]["valid"]
    self.parser = Parser(root=self.datadir,
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
    # only valid set
    # zht:开启全部dropout已得到uncertainty
    self.model.eval()
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()
      for i, (proj_in, proj_mask, pro_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(self.parser.get_valid_set()):
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

        proj_output, _, _, _, _ = self.model(proj_in, proj_mask) #[1,20,64,2048]
        proj_argmax = proj_output[0].argmax(dim=0)  # predict_label [64,2048] 
        # 关键：转变为原始点云
        unproj_argmax = proj_argmax[p_y,p_x] #zht 强制不开启KNN后处理
        if torch.cuda.is_available():
         torch.cuda.synchronize()
        print("Infered seq", path_seq, "scan", path_name,"in", time.time() - end, "sec")
        end = time.time()
        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)
        unproj_label=pro_labels[0][p_y,p_x]

        # save pd gt prob au eu
        path_name1=path_name.split('.')[0]+".pd"
        path1 = os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name1)
        pred_np.tofile(path1)

        unproj_label=unproj_label.cpu().numpy().reshape((-1)).astype(np.int32)
        path_name2=path_name.split(".")[0]+".gt"
        path2=os.path.join(self.logdir, "sequences",
                            path_seq, "predictions", path_name2)
        unproj_label.tofile(path2) #0-19的真值
      print("Finish infering")
        

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./demo.py")
  parser.add_argument(  # 可以考虑换sample /home/zht/Datasets/Semantic_sample
      '--dataset', '-d',
      type=str,
      default = "/home/zht/Datasets/Semantic_sample/", 
      help='Dataset to sample'
  )
  parser.add_argument(
      '--log', '-l',  # 选择预测结果存放路径
      type=str,
      default=  '../../../predict_output/',
      help='Directory to put the predictions. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--model', '-m',
      type=str,
      required=True,
      default=None,
      help='Directory to get the trained model.'
  )
  parser.add_argument(
    '--mode',
    type=str,
    default='valid_split',
    help='Only test_split or valid_split or train_split to choose, can be set by pretrain model directory'
  )
  FLAGS, unparsed = parser.parse_known_args()
  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", FLAGS.dataset)
  print("log", FLAGS.log)
  print("model", FLAGS.model)
  print("----------\n")


  try:
    print("Opening arch config file from %s" % FLAGS.model)
    ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    print("Opening data config file from %s" % FLAGS.model)
    DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file in the !! pretrain model directory!!")
    quit()
  mode = FLAGS.mode
  if(mode=="train_split"):
    using_sequences=DATA["split"]["train"]
  elif(mode=="test_split"):
    using_sequences=DATA["split"]["test"]
  elif(mode=="valid_split"):
    using_sequences=DATA["split"]["valid"]
  # create log folder
  try:
    if not os.path.exists(FLAGS.log):
      os.makedirs(FLAGS.log)
    if not os.path.exists(os.path.join(FLAGS.log, "sequences")):
      os.makedirs(os.path.join(FLAGS.log, "sequences"))
    for seq in using_sequences:
      seq = '{0:02d}'.format(int(seq))
      print("using_list",seq)
      if not os.path.exists(os.path.join(FLAGS.log,"sequences",str(seq))):
        os.makedirs(os.path.join(FLAGS.log,"sequences", str(seq)))
      if not os.path.exists(os.path.join(FLAGS.log,"sequences", str(seq), "predictions")):
        os.makedirs(os.path.join(FLAGS.log,"sequences", str(seq), "predictions"))
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()
  # does model folder exist?
  if os.path.isdir(FLAGS.model):
    print("model folder exists! Using model from %s" % (FLAGS.model))
  else:
    print("model folder doesnt exist! Can't infer...")
    quit()
  # create user and infer dataset
  user = Demo(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model,mode)
  user.infer()