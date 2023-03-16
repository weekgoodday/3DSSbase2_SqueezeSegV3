#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import imp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tasks.semantic.postproc.CRF import CRF
import __init__ as booger

class Segmentator(nn.Module):
  def __init__(self, ARCH, nclasses, path=None, path_append="", strict=False):
    super().__init__()
    self.ARCH = ARCH
    self.nclasses = 20
    self.path = path #path是pretrained_path 可以是None 也可以是包含了Backbone segmentation_decoder segmentation_head12345的五路径 save_checkpoint正好完成了存储过程 所以训练完后找logs就可以接续
    self.path_append = path_append
    self.strict = False
  
    bboneModule = imp.load_source("bboneModule",
                                  booger.TRAIN_PATH + '/backbones/' +
                                  self.ARCH["backbone"]["name"] + '.py')
    self.backbone = bboneModule.Backbone(params=self.ARCH["backbone"]) #有一个backbone模块 在最上层 name是SAC 

    # do a pass of the backbone to initialize the skip connections
    xyz = torch.zeros((1, 3, 
                        self.ARCH['dataset']['sensor']['img_prop']['height'],
                        self.ARCH['dataset']['sensor']['img_prop']['width']))
    stub = torch.zeros((1,
                        self.backbone.get_input_depth(),
                        self.ARCH["dataset"]["sensor"]["img_prop"]["height"],
                        self.ARCH["dataset"]["sensor"]["img_prop"]["width"]))

    if torch.cuda.is_available():
      stub = stub.cuda()
      xyz = xyz.cuda()
      self.backbone.cuda()
    _, stub_skips = self.backbone(stub)

    decoderModule = imp.load_source("decoderModule",
                                    booger.TRAIN_PATH + '/tasks/semantic/decoders/' +
                                    self.ARCH["decoder"]["name"] + '.py')
    self.decoder = decoderModule.Decoder(params=self.ARCH["decoder"],
                                         stub_skips=stub_skips,
                                         OS=self.ARCH["backbone"]["OS"],
                                         feature_depth=self.backbone.get_last_depth()) #有一个decoder模块 在decoder的SAC 

    self.head1 = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(256,
                                        self.nclasses, kernel_size=1,
                                        stride=1, padding=0))

    self.head2 = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(256,
                                        self.nclasses, kernel_size=1,
                                        stride=1, padding=0))

    self.head3 = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(128,
                                        self.nclasses, kernel_size=1,
                                        stride=1, padding=0))

    self.head4 = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(64,
                                        self.nclasses, kernel_size=1,
                                        stride=1, padding=0))
    self.head5 = nn.Sequential(nn.Dropout2d(p=ARCH["head"]["dropout"]),
                              nn.Conv2d(32,
                                        self.nclasses, kernel_size=3,
                                        stride=1, padding=1))



    if self.ARCH["post"]["CRF"]["use"]:
      self.CRF = CRF(self.ARCH["post"]["CRF"]["params"], self.nclasses)
    else:
      self.CRF = None

    # train backbone?
    if not self.ARCH["backbone"]["train"]:
      for w in self.backbone.parameters():
        w.requires_grad = False

    # train decoder?
    if not self.ARCH["decoder"]["train"]:
      for w in self.decoder.parameters():
        w.requires_grad = False

    # train head?
    if not self.ARCH["head"]["train"]:
      for w in self.head.parameters():
        w.requires_grad = False

    # train CRF?
    if self.CRF and not self.ARCH["post"]["CRF"]["train"]:
      for w in self.CRF.parameters():
        w.requires_grad = False

    # print number of parameters and the ones requiring gradients
    # print number of parameters and the ones requiring gradients
    weights_total = sum(p.numel() for p in self.parameters())
    weights_grad = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # breakdown by layer
    weights_enc = sum(p.numel() for p in self.backbone.parameters())
    weights_dec = sum(p.numel() for p in self.decoder.parameters())
    weights_head = sum(p.numel() for p in self.head1.parameters())+\
      sum(p.numel() for p in self.head2.parameters())+\
      sum(p.numel() for p in self.head3.parameters())+\
      sum(p.numel() for p in self.head4.parameters())+\
      sum(p.numel() for p in self.head5.parameters())
    print("Param encoder ", weights_enc)
    print("Param decoder ", weights_dec)
    print("Param head ", weights_head)
    if self.CRF:
      weights_crf = sum(p.numel() for p in self.CRF.parameters())
      print("Param CRF ", weights_crf)

    # get weights
    if path is not None:
      # try backbone
      try:
        w_dict = torch.load(path + "/backbone",
                            map_location=lambda storage, loc: storage)
        self.backbone.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model backbone weights")
      except Exception as e:
        print()
        print("Couldn't load backbone, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try decoder
      try:
        w_dict = torch.load(path + "/segmentation_decoder",
                            map_location=lambda storage, loc: storage)
        self.decoder.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model decoder weights")
      except Exception as e:
        print("Couldn't load decoder, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      # try head
      try:
        print(path_append+'./segmentation_head1')
        w_dict = torch.load(path + "/segmentation_head1",
                            map_location=lambda storage, loc: storage)
        self.head1.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e
      try:
        w_dict = torch.load(path+ "/segmentation_head2",
                            map_location=lambda storage, loc: storage)
        self.head2.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e
      try:
        w_dict = torch.load(path + "/segmentation_head3",
                            map_location=lambda storage, loc: storage)
        self.head3.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      try:
        w_dict = torch.load(path+ "/segmentation_head4",
                            map_location=lambda storage, loc: storage)
        self.head4.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e

      try:
        w_dict = torch.load(path + "/segmentation_head5",
                            map_location=lambda storage, loc: storage)
        self.head5.load_state_dict(w_dict, strict=True)
        print("Successfully loaded model head weights")
      except Exception as e:
        print("Couldn't load head, using random weights. Error: ", e)
        if strict:
          print("I'm in strict mode and failure to load weights blows me up :)")
          raise e
    else:
      print("No path to pretrained, using random init.")

  def forward(self, x, mask=None):

    feature, skips = self.backbone(x) #backbone从输入得到features和skips

    y = self.decoder(feature, skips) #decoder输入feature和skips 得到y 还含有0-4
    # y[0]是最后，通道数最少，特征图最大；y[4]是最前（刚被encoder编码完），通道数最多，特征图最小。
      
    z1 = self.head5(y[0])
    z1 = F.softmax(z1,dim=1)

    z2 = self.head4(y[1])
    z2 = F.softmax(z2,dim=1)

    z3 = self.head3(y[2])
    z3 = F.softmax(z3,dim=1)

    z4 = self.head2(y[3])
    z4 = F.softmax(z4,dim=1)

    z5 = self.head1(y[4])
    z5 = F.softmax(z5,dim=1)

    return [z1, z2, z3, z4, z5] #有不同的head，对应y(decoder输出)的不同维度(不同层数)

  def save_checkpoint(self, logdir, suffix=""):
    # Save the weights
    torch.save(self.backbone.state_dict(), logdir +
               "/backbone" + suffix)
    
    torch.save(self.decoder.state_dict(), logdir +
               "/segmentation_decoder" + suffix)

    torch.save(self.head1.state_dict(),logdir+"/segmentation_head1"+suffix)
    torch.save(self.head2.state_dict(),logdir+"/segmentation_head2"+suffix)
    torch.save(self.head3.state_dict(),logdir+"/segmentation_head3"+suffix)
    torch.save(self.head4.state_dict(),logdir+"/segmentation_head4"+suffix)
    torch.save(self.head5.state_dict(),logdir+"/segmentation_head5"+suffix)

