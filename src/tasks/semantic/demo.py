
#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# import sys
# sys.path.append("github_play/SqueezeSegV3/src/")
import os
os.chdir("/home/zht/github_play/SqueezeSegV3/src/tasks/semantic/")
import argparse
import subprocess
import datetime
import yaml
from shutil import copyfile
import os
import shutil
import __init__ as booger
print(os.path.abspath("./"))
from tasks.semantic.modules.demo_user import *


if __name__ == '__main__':
  parser = argparse.ArgumentParser("./demo.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      default = "../../../sample_data/sequences/", 
      help='Dataset to sample'
  )
  parser.add_argument(
      '--log', '-l',
      type=str,
      default=  '../../../sample_output/',
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
    default='test_split',
    help='Only test_split or train_split to choose, if choose test_split, use sequence08, choose train_split, use sequence 00'
  )
  FLAGS, unparsed = parser.parse_known_args()
  # print summary of what we will do
  print("----------")
  print("INTERFACE:")
  print("dataset", FLAGS.dataset)
  print("log", FLAGS.log)
  print("model", FLAGS.model)
  print("----------\n")


  # open arch config file
  try:
    print("Opening arch config file from %s" % FLAGS.model)
    ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening arch yaml file.")
    quit()

  # open data config file
  try:
    print("Opening data config file from %s" % FLAGS.model)
    DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
  except Exception as e:
    print(e)
    print("Error opening data yaml file.")
    quit()
  mode = FLAGS.mode
  if(mode=="train_split"):
    using_sequences=DATA["split"]["sample"]
  elif(mode=="test_split"):
    using_sequences=DATA["split"]["sample2"]
  # create log folder
  try:
    # if os.path.isdir(FLAGS.log):
    #   shutil.rmtree(FLAGS.log)
    if not os.path.exists(FLAGS.log):
      os.makedirs(FLAGS.log)
    if not os.path.exists(os.path.join(FLAGS.log, "sequences")):
      os.makedirs(os.path.join(FLAGS.log, "sequences"))

    
    for seq in using_sequences:
      seq = '{0:02d}'.format(int(seq))
      print("sample_list",seq)
      if not os.path.exists(os.path.join(FLAGS.log,"sequences",str(seq))):
        os.makedirs(os.path.join(FLAGS.log,"sequences", str(seq)))
      if not os.path.exists(os.path.join(FLAGS.log,"sequences", str(seq), "predictions")):
        os.makedirs(os.path.join(FLAGS.log,"sequences", str(seq), "predictions"))

  except Exception as e:
    print(e)
    print("Error creating log directory. Check permissions!")
    raise

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
  user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model,mode)
  user.infer()
