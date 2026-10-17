import os
import glob
import numpy as np

folders = ['yifu','floor1','floor2','floor3','floor4','floor5','floor8','floor10','202a','311a','311a_2','307','307_2','308','308_2','dahuof1','dahuof2','dormitory323','library','dahuo306','dacheng']
count = 0
for dir in folders:
   files = glob.glob('./'+dir+'/nir_rgb/label_gt/*.npy')
   for file in files:
      gt = np.load(file)
      h,w = gt.shape
      if (np.sum(gt != 0)/(h*w))>0.1:
        count += 1

print(count)
