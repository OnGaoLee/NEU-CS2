import os
import glob
import numpy as np

folders = ['sec1','sec2','sec3','sec4','sec5','sec6','sec7','sec8','sec9','sec10','sec11','sec12']
count = 0
for dir in folders:
   files = glob.glob('./'+dir+'/nir_rgb/label_gt/*.npy')
   for file in files:
      gt = np.load(file)
      h,w = gt.shape
      if (np.sum(gt != 0)/(h*w))>0.1:
        count += 1

print(count)
