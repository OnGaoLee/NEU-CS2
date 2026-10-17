import numpy as np
import cv2
import math
import sys
import os
import glob
import argparse
import shutil

current_file = os.path.abspath(__file__).split('/')[-1]
current_path = os.path.abspath(__file__).replace(current_file,'')
# print(current_path)
sys.path.append(current_path.replace("/MS2", "")) 
sys.path.append(current_path) 
sys.path.append("..") 
sys.path.append(".") 
# print(sys.path)
from label_filter_neucs2 import label_filter, get_gradient_confidence, get_consistency_confidence
from dataset_utils import disp_transform, get_fov

def load_calib_data(path):
  
  calib_data = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

  # nirL data
  K_nirL = calib_data.getNode("K_nirL").mat()
  D_nirL = calib_data.getNode("D_nirL").mat()
  H_nirL = calib_data.getNode("H_nirL").real()
  W_nirL = calib_data.getNode("W_nirL").real()

  # nirR data
  K_nirR = calib_data.getNode("K_nirR").mat()
  D_nirR = calib_data.getNode("D_nirR").mat()

  # rgbR data
  K_rgbR = calib_data.getNode("K_rgbR").mat()
  D_rgbR = calib_data.getNode("D_rgbR").mat()
  H_rgbR = calib_data.getNode("H_rgbR").real()
  W_rgbR = calib_data.getNode("W_rgbR").real()


  # nirL to rgbR transformation matrix
  R_nir2rgb = calib_data.getNode("R_nir2rgb").mat()
  T_nir2rgb = calib_data.getNode("T_nir2rgb").mat()

  # nirL to nirR transformation matrix
  T_nir2nir = calib_data.getNode("T_nir2nir").mat()

  # baseline
  baseline_nirrgb = np.linalg.norm(T_nir2rgb)
  baseline_nirnir = -T_nir2nir[0,0]

  return K_rgbR, D_rgbR, H_rgbR, W_rgbR, baseline_nirnir , K_nirL, D_nirL, H_nirL, W_nirL, baseline_nirrgb, R_nir2rgb, T_nir2rgb


def showRectify(unrectifyL, unrectifyR, rectifyL, rectifyR):

    rectify_size = rectifyL.shape
    unrectifyL = cv2.resize(unrectifyL,(rectify_size[1], rectify_size[0]))
    unrectifyR = cv2.resize(unrectifyR,(rectify_size[1], rectify_size[0]))
    unrectifyLR = cv2.hconcat([unrectifyL, unrectifyR])
    rectifyLR = cv2.hconcat([rectifyL, rectifyR])

    cv2.rectangle(rectifyLR, (0,0), (rectify_size[1],rectify_size[0]),(255,0,0),3)
    cv2.rectangle(rectifyLR, (rectify_size[1],0), (2*rectify_size[1], rectify_size[0]),(255,0,0),3)

    mixLR = cv2.vconcat([unrectifyLR,rectifyLR])

    h, w, _ = mixLR.shape

    scalar = 1200/max(h, w) # 缩放图像宽度为1200，方便可视化
    H = int(h * scalar)
    W = int(w * scalar)
    scaleMixLR = cv2.resize(mixLR, (W,H), interpolation=cv2.INTER_LINEAR)

    for i in range(0, H, 16):
        cv2.line(scaleMixLR, (0,i), (W,i), (0,255,0))

    cv2.namedWindow("result",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result",scaleMixLR)
    key = cv2.waitKey(0)
    if key == 27:
      sys.exit()
    elif key == ord('s'):
      cv2.imwrite("./rectify_result.png",scaleMixLR)


  
def rename_files(path, prefix):

  # 遍历文件夹中的所有文件
  for filename in os.listdir(path):
      # 构造新的文件名，这里以在原文件名前加上前缀作为示例
      new_filename = prefix + filename.split('_')[-1]
    
      # 构造完整的文件路径
      old_filepath = os.path.join(path, filename)
      new_filepath = os.path.join(path, new_filename)
    
      # 重命名文件
      os.rename(old_filepath, new_filepath)

  print("rename done！")
    

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='Preprocess MS2 dataset')
  parser.add_argument('--input_path', type=str, required=False, default="/media/jiayu/share/win10data/datasets/MS2/sync_data/_2021-08-13-15-46-56", help='Path to load MS2 dataset')
  parser.add_argument('--output_path', type=str, required=False, default="/home/jiayu/Desktop/datasets/MS2_processed", help='Path to save processed MS2 dataset')
  parser.add_argument('--stereo_type', type=str, required=False, default="cre", choices=['cre','psm','sgbm','igev'], help='stereo method of pesudo label generation')
  parser.add_argument('--type', type=str, required=False, default="train", choices=["train", "test"],help='choose data type. train / test / both')
  parser.add_argument('--scale', type=float, required=False, default=0.5, help='set a resize scale of rectified images')
  parser.add_argument('--no_filter', action='store_true', help='set a resize scale of rectified images')
  parser.add_argument('--occlusion_mask', action='store_true', help='set a resize scale of rectified images')

  args = parser.parse_args()

  neucs2_path = args.input_path
  output_path = args.output_path
  label_dirs = {'cre':'label_cre','igev':'label_igev','psm':'label_psm','sgbm':'label_sgbm'}
  label_dir = label_dirs[args.stereo_type]

  # ================load calib data===================
  calib_data_file = neucs2_path + "/calib_ori.xml"
  calib_data = load_calib_data(calib_data_file)

  K_rgbR = calib_data[0]
  D_rgbR = calib_data[1]
  baseline_nirnir = calib_data[4]
  K_nirL = calib_data[5]
  D_nirL = calib_data[6]
  baseline_nirrgb = calib_data[9]
  R_nir2rgb = calib_data[10]
  T_nir2rgb = calib_data[11]


  H_nirL = calib_data[7]
  W_nirL = calib_data[8]

  H_rgbR = calib_data[2]
  W_rgbR = calib_data[3]

  # ===================compute FOV=====================
  # nir_hfov, nir_wfov = get_fov(H_nirL, W_nirL, K_nirL[0,0])
  # rgb_hfov, rgb_wfov = get_fov(H_rgbR, W_rgbR, K_rgbR[0,0])

  # nir:(0.54, 1.57) rgb:(0.49, 1.35)
  # print(nir_hfov, nir_wfov, rgb_hfov, rgb_wfov)

  if args.type in ["train", "test"]:
    place_name = neucs2_path.split("/")[-2]
    scene_name = neucs2_path.split("/")[-1]
    train_path = output_path + "/" + args.type + '/' + place_name
    os.makedirs(train_path, exist_ok=True)
    nirrgb_datapath = train_path + "/nir_rgb"
    nirnir_datapath = train_path + "/nir_nir"
    # ==================RGB-NIR rectification===========

    imageSize =(int(W_nirL), int(H_nirL))
    s = args.scale # 校正后图像分辨率的缩放尺度
    newImageSize = (int(W_nirL*s),int(H_nirL*s)) # 校正后图像分辨率
    print(imageSize,newImageSize)

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(K_nirL, D_nirL, K_rgbR, D_rgbR, imageSize, R_nir2rgb, T_nir2rgb, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0, newImageSize=newImageSize)

    rectify_mono_l = np.dot(np.dot(PL[:,:3], RL), np.linalg.inv(K_nirL))
    rectify_mono_R = np.dot(np.dot(PR[:,:3], RR), np.linalg.inv(K_rgbR))

    rgb_datapath = neucs2_path + "/rgb"
    nir_datapath = neucs2_path + "/nir_nir"

    os.makedirs(nirrgb_datapath + "/left", exist_ok=True)
    os.makedirs(nirrgb_datapath + "/right", exist_ok=True)

    ## ======save new calib data==================
    calib_path = train_path + '/calibration_data'
    os.makedirs(calib_path, exist_ok=True)
    if not os.path.exists(calib_path + '/calib_nir_rgb.xml'):
      calib_nir_rgb = cv2.FileStorage(calib_path + '/calib_nir_rgb.xml', cv2.FILE_STORAGE_WRITE)
      calib_nir_rgb.write('mtxL', PL[:,:3])
      calib_nir_rgb.write('rectL', RL)
      calib_nir_rgb.write('h', H_nirL*s)
      calib_nir_rgb.write('w', W_nirL*s)
      calib_nir_rgb.write('baseline', baseline_nirrgb*1000)
      calib_nir_rgb.release()
    
    if not os.path.exists(calib_path + '/calib_nir_nir.xml'):
      calib_rgb_rgb = cv2.FileStorage(calib_path + '/calib_nir_nir.xml', cv2.FILE_STORAGE_WRITE)
      calib_rgb_rgb.write('mtxL', K_nirL)
      calib_rgb_rgb.write('rectL', np.eye(3))
      calib_rgb_rgb.write('h', H_nirL)
      calib_rgb_rgb.write('w', W_nirL)
      calib_rgb_rgb.write('baseline', baseline_nirnir*1000)
      calib_rgb_rgb.release()

    print('generate now calibration data done!')

    left_files = sorted(glob.glob(nir_datapath + "/left/*.png"))
    if args.type == 'train':
      if os.path.exists(nirnir_datapath+'/left'):
        shutil.rmtree(nirnir_datapath+'/left')
        os.makedirs(nirnir_datapath+'/left')
      else:
        os.makedirs(nirnir_datapath+'/left',exist_ok=True)
      if os.path.exists(nirnir_datapath+'/right'):
        shutil.rmtree(nirnir_datapath+'/right')
        os.makedirs(nirnir_datapath+'/right')
      else:
        os.makedirs(nirnir_datapath+'/right',exist_ok=True)

      for left_file in left_files:
        right_file = left_file.replace('left', 'right')
        shutil.copyfile(left_file, left_file.replace(nir_datapath,nirnir_datapath))
        shutil.copyfile(right_file, right_file.replace(nir_datapath,nirnir_datapath))
      if os.path.exists(nirnir_datapath + "/left"):
        rename_files(nirnir_datapath + "/left", scene_name+'_')      
      if os.path.exists(nirnir_datapath + "/right"):
        rename_files(nirnir_datapath + "/right", scene_name+'_')      
    else:
      os.makedirs(nirnir_datapath+'/left',exist_ok=True)
      os.makedirs(nirnir_datapath+'/right', exist_ok=True)
      for left_file in left_files:
        right_file = left_file.replace('left', 'right')
        shutil.copyfile(left_file, left_file.replace(nir_datapath,nirnir_datapath))
        left_old_name = left_file.replace(nir_datapath,nirnir_datapath)
        left_new_name = left_old_name.replace('0.png','')+scene_name+'_0.png'
        os.rename(left_old_name, left_new_name)
        shutil.copyfile(right_file, right_file.replace(nir_datapath,nirnir_datapath))
        right_old_name = right_file.replace(nir_datapath,nirnir_datapath)
        right_new_name = right_old_name.replace('0.png','')+scene_name+'_0.png'
        os.rename(right_old_name, right_new_name)

      

    print("copy nir-nir origin image done!")

    nir_left_files = sorted(glob.glob(nir_datapath + "/left/*.png"))

    for nir_left_file in nir_left_files:
      file_name = nir_left_file.split("/")[-1]
      if os.path.exists(nirrgb_datapath+"/right/"+scene_name + '_' +file_name) & (os.path.exists(nirrgb_datapath+"/left/"+scene_name + '_' +file_name)):
        continue
      nir_right_file = nir_left_file.replace("left", "right")
      rgb_file = nir_left_file.replace("nir_nir/left", "rgb")

      unrectifyL = cv2.imread(nir_left_file)
      unrectifyR = cv2.imread(rgb_file)
    ## ===== 调用opencv函数initUndistortRectifyMap ======

      # # 校正映射
      # mapLx, mapLy = cv2.initUndistortRectifyMap(K_nirL, D_nirL, RL, PL, newImageSize, cv2.CV_32FC1)
      mapRx, mapRy = cv2.initUndistortRectifyMap(K_rgbR, D_rgbR, RR, PR, newImageSize, cv2.CV_32FC1)

      # rectifyL = cv2.remap(unrectifyL, mapLx, mapLy, cv2.INTER_LINEAR)
      rectifyR = cv2.remap(unrectifyR, mapRx, mapRy, cv2.INTER_LINEAR)

      ## ===== 调用opencv函数warpPerspective ======

      rectifyL = cv2.warpPerspective(unrectifyL, rectify_mono_l, newImageSize)
      # rectifyR = cv2.warpPerspective(unrectifyR, rectify_mono_R, newImageSize)

      cv2.imwrite(nirrgb_datapath+"/left/"+scene_name + '_' + file_name, rectifyL)
      cv2.imwrite(nirrgb_datapath+"/right/"+scene_name + '_' +file_name, rectifyR)

      # showRectify(unrectifyL,unrectifyR,rectifyL,rectifyR)


    print("generate nir-rgb pairs done!")
    # sys.exit()

    # ==============nirnir depth gt to nirnir and nirrgb disp gt===============

    depth_gt_path = neucs2_path + "/depth"

    new_depth_gt_path = nirnir_datapath + "/label_gt"
    new_depth_gt_nirrgb_path = nirrgb_datapath + "/label_gt"
    os.makedirs(new_depth_gt_nirrgb_path,exist_ok=True)
    os.makedirs(new_depth_gt_path,exist_ok=True)


    depth_gt_files = glob.glob(depth_gt_path + "/*.npy", recursive=False)

    depth_mtx = np.dot(np.dot(RL, np.linalg.inv(np.eye(3))), np.linalg.inv(K_nirL))

    for depth_gt_file in depth_gt_files:
      depth_gt_name = depth_gt_file.split("/")[-1]
      if os.path.exists(nirrgb_datapath + "/label_gt/" +scene_name + '_' + depth_gt_name):
        continue
        pass
      depth_gt = np.load(depth_gt_file)
      if args.type == 'test':
        depth_gt = np.float32((depth_gt > 0.6) & (depth_gt<6))*depth_gt
      else:
        depth_gt = np.float32((depth_gt > 0.6) & (depth_gt<10))*depth_gt
      disp_gt = np.float32(depth_gt > 0)*K_nirL[0,0]*baseline_nirnir/(depth_gt+1e-6)
      disp_cre_file = depth_gt_file.replace('depth','label_cre')
      if os.path.exists(disp_cre_file):
        disp_cre = np.load(disp_cre_file)


      if os.path.exists(new_depth_gt_path + '/' +scene_name + '_' + depth_gt_name):
        continue
        pass

      np.save(new_depth_gt_path + '/' +scene_name + '_' + depth_gt_name, disp_gt)
      left_img_file =depth_gt_file.replace('depth','/nir_nir/left').replace('npy','png')
      right_img_file = depth_gt_file.replace('depth','nir_nir/right').replace('npy','png')
      disp_gt_filter = label_filter(left_img_file,right_img_file, new_depth_gt_path + '/' +scene_name+'_'+ depth_gt_name, show=False, delta_g=4, delta_r=1)
      mask_c = np.ones_like(disp_gt_filter)
      if os.path.exists(disp_cre_file):
        _, mask_c = get_consistency_confidence(disp_gt_filter,disp_cre,delta_c=1.5)
      disp_gt_filter = np.float32(mask_c)*disp_gt_filter

      np.save(new_depth_gt_path + '/' +scene_name + '_' + depth_gt_name, disp_gt_filter)
      disp_gt_im = np.uint8(disp_gt_filter/disp_gt_filter.max()*255)
      disp_gt_color = cv2.applyColorMap(disp_gt_im, cv2.COLORMAP_INFERNO)

      disp_nirrgb = disp_transform(disp_gt_filter, K_nirL[0,0], baseline_nirnir, PL[0,0], baseline_nirrgb, (newImageSize[1],newImageSize[0]), depth_mtx, rectify_mono_l, depth=False, forward_mapping=False)

      _, mask_g = get_gradient_confidence(disp_nirrgb, 1/(disp_nirrgb+1e-6), delta_g=2.5)
      np.save(nirrgb_datapath + "/label_gt/" +scene_name + '_' + depth_gt_name, np.float32(mask_g)*disp_nirrgb)

      disp_nirrgb_im = np.uint8(disp_nirrgb/disp_nirrgb.max()*255)
      disp_nirrgb_color = cv2.applyColorMap(disp_nirrgb_im, cv2.COLORMAP_INFERNO)
      cv2.imwrite(nirrgb_datapath + "/label_gt/" +scene_name + '_' + depth_gt_name.replace(".npy",".png"), disp_nirrgb_color)
      cv2.imwrite(nirnir_datapath + "/label_gt/" +scene_name + '_' + depth_gt_name.replace(".npy",".png"), disp_gt_color)

      # cv2.imshow("disprgbnir", disp_nirrgb_color)
      # cv2.imshow("depthgt", disp_gt_color)
      # cv2.waitKey(0)

    print("generate nir-rgb disp gt done!")
    # sys.exit()

    # ==============rgbrgb disp pesudo to rgbnir disp pesudo===============

    disp_cre_path = nir_datapath.replace('nir_nir', '') + "/"+ label_dir
    if not args.no_filter:
      new_disp_cre_path = nirrgb_datapath + "/" + label_dir
    else:
      new_disp_cre_path = nirrgb_datapath + "/" + label_dir + "_unfiltered"
    if args.occlusion_mask:
      new_occ_mask_path = nirrgb_datapath + "/occ_mask"
      os.makedirs(new_occ_mask_path, exist_ok=True)
      
    os.makedirs(new_disp_cre_path, exist_ok=True)

    disp_cre_files = glob.glob(disp_cre_path + "/*.npy", recursive=False)
    if len(disp_cre_files) == 0:
      print("Please generate pesudo label first.")
      sys.exit()

    for disp_cre_file in disp_cre_files:
      rgb_l_file = disp_cre_file.replace(label_dir, "nir_nir/left").replace("npy", "png")
      rgb_r_file = disp_cre_file.replace(label_dir, "/nir_nir/right").replace("npy", "png")
      disp_cre_name = disp_cre_file.split("/")[-1]
      if os.path.exists(new_disp_cre_path+"/" +scene_name + '_' + disp_cre_name):
        if not args.occlusion_mask:
          continue
        elif os.path.exists(new_occ_mask_path+"/" + scene_name + '_' +disp_cre_name):
          continue
      disp_cre = np.load(disp_cre_file)
      if not args.no_filter:
        disp_cre_filtered = label_filter(rgb_l_file,rgb_r_file, disp_cre_file, show=False,delta_g=4,delta_r=0.2)

        disp_nirrgb = disp_transform(disp_cre_filtered, K_nirL[0,0], baseline_nirnir, PL[0,0], baseline_nirrgb, (newImageSize[1],newImageSize[0]), depth_mtx, rectify_mono_l, depth=False, forward_mapping=False)
        _, mask_g = get_gradient_confidence(disp_nirrgb, 1/(disp_nirrgb+1e-6), delta_g=2.5)

        if not os.path.exists(new_disp_cre_path+"/" +scene_name + '_' + disp_cre_name):
          np.save(new_disp_cre_path+"/"+scene_name + '_' +disp_cre_name, disp_nirrgb*mask_g)
        if args.occlusion_mask:
          occ_mask = np.logical_and((disp_nirrgb !=0), (mask_g !=0 ))
          np.save(new_occ_mask_path + "/" +scene_name + '_'+ disp_cre_name, occ_mask)
          occ_mask_im = np.uint8(np.uint8(occ_mask)*255)
          cv2.imwrite(new_occ_mask_path + "/" +scene_name + '_'+ disp_cre_name.replace("npy", "png"), occ_mask_im)
      else:
        disp_nirrgb = disp_transform(disp_cre, K_nirL[0,0], baseline_nirnir, PL[0,0], baseline_nirrgb, (newImageSize[1],newImageSize[0]), depth_mtx, rectify_mono_l, depth=False, forward_mapping=False)
        _, mask_g = get_gradient_confidence(disp_nirrgb, disp_nirrgb, delta_g=10)
        
        mask_g = np.ones_like(mask_g) # 如果要剔除边缘就注释这一行代码
        disp_nirrgb = disp_nirrgb*mask_g
        np.save(new_disp_cre_path+"/"+scene_name + '_' +disp_cre_name, disp_nirrgb)
        

      disp_nirrgb_im = np.uint8(np.clip(disp_nirrgb/np.percentile(disp_nirrgb,99),0,1)*255)
      disp_nirrgb_color = cv2.applyColorMap(disp_nirrgb_im, cv2.COLORMAP_INFERNO)

      cv2.imwrite(new_disp_cre_path+"/"+scene_name + '_' +disp_cre_name.replace("npy", "png"), disp_nirrgb_color)

      # cv2.imshow("disprgbnir", disp_nirrgb_color)
      # cv2.waitKey(0)

    print("generate rgb-nir disp cre done!")
    if args.occlusion_mask:
      print("generate rgb-nir occlusion mask done!")
