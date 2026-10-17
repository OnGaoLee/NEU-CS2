import os
import glob
import numpy as np

folders = ['yifu','floor1','floor2','floor3','floor4','floor5','floor8','floor10','202a','311a','307','308','dahuof1','dahuof2','dormitory323','library','dahuo306','dacheng','mix']

count = 0
for dir in folders:
   files = glob.glob('./'+dir+'/nir_rgb/label_gt/*.npy')
   for npy_file in files:
      gt = np.load(npy_file)
      h,w = gt.shape
      if (np.sum(gt != 0)/(h*w))<0.5:
        count += 1
        npy_im_file = npy_file.replace('npy','png')
        cre_npy_file = npy_file.replace('label_gt','label_cre')
        cre_im_file = npy_im_file.replace('label_gt','label_cre')
        u_cre_npy_file = npy_file.replace('label_gt','label_cre_unfiltered')
        u_cre_im_file = npy_im_file.replace('label_gt','label_cre_unfiltered')
        mask_npy_file = npy_file.replace('label_gt','occ_mask')
        mask_im_file = npy_im_file.replace('label_gt','occ_mask')
        nir_file = npy_im_file.replace('label_gt','left')
        rgb_file = npy_im_file.replace('label_gt','right')

        nir_l_file = npy_im_file.replace('nir_rgb/label_gt','nir_nir/left')
        nir_r_file = npy_im_file.replace('nir_rgb/label_gt','nir_nir/right')
        nir_gt_im_file = npy_im_file.replace('nir_rgb/label_gt','nir_nir/label_gt')
        nir_gt_file = nir_gt_im_file.replace('png','npy')
        if os.path.exists(npy_file):
          os.remove(npy_file)
        if os.path.exists(npy_im_file):
          os.remove(npy_im_file)
        if os.path.exists(cre_im_file):
          os.remove(cre_im_file)
        if os.path.exists(cre_npy_file):
          os.remove(cre_npy_file)
        if os.path.exists(u_cre_npy_file):
          os.remove(u_cre_npy_file)
        if os.path.exists(u_cre_im_file):
          os.remove(u_cre_im_file)
        if os.path.exists(mask_im_file):
          os.remove(mask_im_file)
        if os.path.exists(mask_npy_file):
          os.remove(mask_npy_file)
        if os.path.exists(nir_file):
          os.remove(nir_file)
        if os.path.exists(rgb_file):
          os.remove(rgb_file)
        if os.path.exists(nir_l_file):
          os.remove(nir_l_file)
        if os.path.exists(nir_r_file):
          os.remove(nir_r_file)
        if os.path.exists(nir_gt_file):
          os.remove(nir_gt_file)
        if os.path.exists(nir_gt_im_file):
          os.remove(nir_gt_im_file)
print("remove num: ", count)
