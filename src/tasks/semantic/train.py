#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
import wandb
print("append path:",os.path.abspath(os.path.join(os.path.abspath(__file__),"../../..")))
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.abspath(__file__),"../../..")))
from tasks.semantic.modules.trainer import *
#os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./train.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      default='/home/zht/Datasets/Semantic',
      #required=True,
      help='Dataset to train with. No Default',
  )
  parser.add_argument( 
      '--arch_cfg', '-ac',
      type=str,
      #required=True,
      default='/home/zht/github_play/SqueezeSegV3/src/tasks/semantic/config/arch/SSGV321.yaml',
      help='Architecture yaml cfg file. See /config/arch for sample. No default!',
  )
  parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      required=False,
      default='/home/zht/github_play/SqueezeSegV3/src/tasks/semantic/config/labels/semantic-kitti.yaml',
      help='Classification yaml cfg file. See /config/labels for sample. No default!',
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=os.path.expanduser("~") + '/logs/' +
      datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
      help='Directory to put the log data. Default: ~/logs/date+time'
  )
  parser.add_argument(
      '--pretrained', '-p',
      type=str,
      required=False,
      default=None,
      help='Directory to get the pretrained model. If not passed, do from scratch!'
  )
  parser.add_argument(
      '--name', 
      type=str,
      required=False,
      default=None,
      help='wandbname if not None, use wandb'
  )
  parser.add_argument(
      '--dropout_rate',
      type=float,
      default=0.01,
      #required=True,
      help='if change dropout rate to train, use this parameter',
  )
  FLAGS, unparsed = parser.parse_known_args()
  wandb_open=False
  if(FLAGS.name != None):
    wandb_open=True
  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", FLAGS.dataset)
  print("arch_cfg", FLAGS.arch_cfg)
  print("data_cfg", FLAGS.data_cfg)
  print("log", FLAGS.log)
  print("pretrained", FLAGS.pretrained)
  print("----------\n")

  # open arch config file
  try:
    print("Opening arch config file %s" % FLAGS.arch_cfg)
    ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    print("Opening data config file %s" % FLAGS.data_cfg)
    DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()

  # create log folder
  try:
    if os.path.isdir(FLAGS.log):
      shutil.rmtree(FLAGS.log)
    os.makedirs(FLAGS.log)
  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    quit()

  # does model folder exist?
  if FLAGS.pretrained is not None:
    if os.path.isdir(FLAGS.pretrained):
      print("model folder exists! Using model from %s" % (FLAGS.pretrained))
    else:
      print("model folder doesnt exist! Start with random weights...")
  else:
    print("No pretrained directory found.")

  # copy all files to log folder (to remember what we did, and make inference
  # easier). Also, standardize name to be able to open it later
  try:
    print("Copying files to %s for further reference." % FLAGS.log)
    copyfile(FLAGS.arch_cfg, FLAGS.log + "/arch_cfg.yaml")
    copyfile(FLAGS.data_cfg, FLAGS.log + "/data_cfg.yaml")
  except Exception as e:
    print(e)
    print("Error copying files, check permissions. Exiting...")
    quit()
  if(wandb_open):
    wandb.init(project="my_ssv3_project",name=FLAGS.name)
  # # create trainer and start the training
  trainer = Trainer(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.pretrained,wandb_open,FLAGS.dropout_rate)
  trainer.train()
