import sys
import os
import cv2
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

def disp2im(disp):
    disp = np.abs(disp)
    im = np.clip((disp-disp.min()) / (disp.max()-np.min(disp)), 0, 1)
    return im

def get_rgb(file):

    rgb = Image.open(file)
    rgb.convert('RGB')

    return rgb

def distance(x, d_min=0.2, d_max=192, delta_d_min=0.1, delta_d_max=0.5):
  """计算距离置信度
  Args:
      x : 输入
      d_min (_type_): 可信深度区间左边界
      d_max (_type_): 可信深度区间右边界
      delta_d_min (_type_): 系数
      delta_d_max (_type_): 系数
  """

  if x < d_min:
      result = np.exp(-np.abs(x - d_min)**2/(2*delta_d_min**2))
  elif x > d_max:
      result = np.exp(-np.abs(x - d_max)**2/(2*delta_d_max**2))
  else:
      result = 1
  return result

def get_distance_confidence(label, d_min=0.2, d_max=192):
  """获取距离置信度，及其掩码

  Args:
      label (_type_): 视差图
  """
  w_d = np.vectorize(distance)(label, d_min, d_max)
  mask = np.where(w_d < 0.5, 0, 1)
  return w_d, mask

def gradient(x, delta_g=2.5):
  """计算梯度置信度"""
  return np.exp(-x**2/(2*delta_g**2))
    
def get_gradient_confidence(label, label_reverse, delta_g=2.5):
  """获取梯度置信度，及其掩码

  Args:
      label (_type_): 单目转双目视差图
      label_reverse (_type_): 单目转双目逆视差图
  """
  gradient_magnitudes = []
  gradient_magnitudes_reverse = []
  sobel_x = cv2.Sobel(label, cv2.CV_64F, 1, 0, ksize=3)  # 在x方向上计算梯度
  sobel_y = cv2.Sobel(label, cv2.CV_64F, 0, 1, ksize=3)  # 在y方向上计算梯度
  sobel_x_reverse = cv2.Sobel(label_reverse, cv2.CV_64F, 1, 0, ksize=3)  # 在x方向上计算梯度
  sobel_y_reverse = cv2.Sobel(label_reverse, cv2.CV_64F, 0, 1, ksize=3)  # 在y方向上计算梯度
  # 计算梯度的幅值和方向
  gradient_magnitude = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
  gradient_magnitude_reverse = np.sqrt(np.square(sobel_x_reverse) + np.square(sobel_y_reverse))

  w_g_forward = np.vectorize(gradient)(gradient_magnitude, delta_g)
  w_g_reverse = np.vectorize(gradient)(gradient_magnitude_reverse, delta_g)
  w_g = w_g_forward * w_g_reverse
  mask = np.where(w_g < 0.5, 0, 1)

  return w_g, mask

def stereo_image_warp(img, disp, sign=-1):
    "2d->2d图像变换,利用双目视差"
    h, w = disp.shape[-2:]
    theta = torch.Tensor(np.array([[1, 0, 0], [0, 1, 0]])).to(disp.device)
    theta = theta.expand((disp.size()[0], 2, 3)).contiguous()
    grid = F.affine_grid(theta, disp.size())
    disp = disp.transpose(1, 2).transpose(2, 3)
    disp = torch.cat((disp, torch.zeros(disp.size()).to(disp.device)), 3)

    # 确保grid取值范围是(2j/w-1, 2i/h-1)=(-1,1)
    grid = grid + 2 * disp * sign / disp.shape[2]
    sampled = F.grid_sample(img.float(), grid.float())
    return sampled

def get_reprojection_confidence(label, left_img, right_img, delta_r=0.25):
  """计算重投影置信度,剔除遮挡区域掩码时delta_r=0.2,保留较多有效值时delta_r=0.75

  Args:
      label (_type_): _description_
      left_img (_type_): _description_
      right_img (_type_): _description_
      delta_r (float, optional): _description_. Defaults to 0.5.
  """
  label_t = torch.tensor(label).unsqueeze(0).unsqueeze(0)
  left_img_warped = stereo_image_warp(right_img, label_t)
  diff = torch.mean(torch.abs(left_img - left_img_warped), dim=1, keepdim=True)
  w_r = torch.exp(-diff**2/(2*delta_r**2)).squeeze(0).squeeze(0).cpu().numpy()
  mask = np.where(w_r < 0.5, 0, 1)

  return w_r, mask
  
def get_consistency_confidence(label1, label2, delta_c=0.25):
  """计算一致性置信度,将真值中的异常值过滤

  Args:
      label (_type_): _description_
      left_img (_type_): _description_
      right_img (_type_): _description_
      delta_r (float, optional): _description_. Defaults to 0.5.
  """
  label1_t = torch.tensor(label1).unsqueeze(0).unsqueeze(0)
  label2_t = torch.tensor(label2).unsqueeze(0).unsqueeze(0)
  diff = torch.mean(torch.abs(label1_t - label2_t), dim=1, keepdim=True)
  w_r = torch.exp(-diff**2/(2*delta_c**2)).squeeze(0).squeeze(0).cpu().numpy()
  mask = np.where(w_r < 0.5, 0, 1)

  return w_r, mask

def generate_confidence_map(disp, left_img, right_img, d_min=1, d_max=192,delta_g=2.5, delta_r=0.2):
    disp_reverse = 1 /(disp + 1e-6)
    w_d, mask_d = get_distance_confidence(disp, d_min, d_max)
    w_g, mask_g = get_gradient_confidence(disp, disp_reverse,delta_g)
    w_r, mask_r = get_reprojection_confidence(disp, left_img, right_img,delta_r)
    confidence_map = w_d * w_g * w_r
    
    return confidence_map, np.float32(mask_d), np.float32(mask_g), np.float32(mask_r)


def label_filter(left_file, right_file, label_file, show=False,delta_g=2.5, delta_r=0.2):

  mean = [0.485, 0.456, 0.406]
  std = [0.229, 0.224, 0.225]
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(std=std, mean =mean)])

  label = np.load(label_file)
  left_img = transform(get_rgb(left_file)).unsqueeze(0)
  right_img = transform(get_rgb(right_file)).unsqueeze(0)

  confidence_map, mask_d, mask_g, mask_r = generate_confidence_map(label, left_img, right_img,delta_g=delta_g, delta_r=delta_r)

  mask_all = mask_d*mask_g*mask_r

  mask_all_im = np.uint8(mask_all * 255.0)
  mask_d_im = np.uint8(mask_d * 255.0)
  mask_g_im = np.uint8(mask_g * 255.0)
  mask_r_im = np.uint8(mask_r * 255.0)

  label_masked_all = label*mask_all

  if show:
      label_masked_d = label*mask_d
      label_masked_g = label*mask_g
      label_masked_r = label*mask_r

      label_img_color = cv2.applyColorMap(np.uint8(disp2im(label) * 255.0), cv2.COLORMAP_INFERNO)
      label_masked_all_color = cv2.applyColorMap(np.uint8(disp2im(label_masked_all) * 255.0), cv2.COLORMAP_INFERNO)
      label_masked_d_color = cv2.applyColorMap(np.uint8(disp2im(label_masked_d) * 255.0), cv2.COLORMAP_INFERNO)
      label_masked_g_color = cv2.applyColorMap(np.uint8(disp2im(label_masked_g) * 255.0), cv2.COLORMAP_INFERNO)
      label_masked_r_color = cv2.applyColorMap(np.uint8(disp2im(label_masked_r) * 255.0), cv2.COLORMAP_INFERNO)

      left = cv2.imread(left_file)
      right = cv2.imread(right_file)
      right_ = transforms.ToTensor()(get_rgb(right_file)).unsqueeze(0)
      label_t = torch.tensor(label).unsqueeze(0).unsqueeze(0)
      left_warped = stereo_image_warp(right_, label_t, -1)
      left_warped = left_warped.squeeze(0).permute(1,2,0).cpu().numpy()
      left_warped_img = cv2.cvtColor(np.uint8(left_warped*255), cv2.COLOR_RGB2BGR)
      left_warped_img_masked = cv2.cvtColor(np.uint8(left_warped*mask_all[:,:,None].repeat(3,axis=-1)*255), cv2.COLOR_RGB2BGR)
    
      cv2.namedWindow('img_vis',cv2.WINDOW_NORMAL)
      cv2.namedWindow('label_vis',cv2.WINDOW_NORMAL)

      images = np.hstack((left, left_warped_img, left_warped_img_masked,right))
      labels = np.hstack((label_img_color, label_masked_all_color, label_masked_d_color, label_masked_g_color, label_masked_r_color))

      cv2.imshow('img_vis', images)
      cv2.imshow('label_vis', labels)
      key = cv2.waitKey(0)
      if key == 27:
        sys.exit()
      elif key == ord('s'):
    
        mask_d_path = label_file.replace('label_cre', 'label_cre_filter/mask_d').replace('npy', 'png')
        mask_g_path = label_file.replace('label_cre', 'label_cre_filter/mask_g').replace('npy', 'png')
        mask_r_path = label_file.replace('label_cre', 'label_cre_filter/mask_r').replace('npy', 'png')
        mask_all_path = label_file.replace('label_cre', 'label_cre_filter/mask_all').replace('npy', 'png')
        label_mask_d_path = label_file.replace('label_cre', 'label_cre_filter/label_mask_d').replace('npy', 'png')
        label_mask_g_path = label_file.replace('label_cre', 'label_cre_filter/label_mask_g').replace('npy', 'png')
        label_mask_r_path = label_file.replace('label_cre', 'label_cre_filter/label_mask_r').replace('npy', 'png')
        label_mask_all_path = label_file.replace('label_cre', 'label_cre_filter/label_mask_all').replace('npy', 'png')
        # np.save(label_file, label_masked_all)
        os.makedirs(mask_d_path.replace('/' + mask_d_path.split('/')[-1],''),exist_ok=True)
        os.makedirs(mask_g_path.replace('/' + mask_g_path.split('/')[-1],''),exist_ok=True)
        os.makedirs(mask_r_path.replace('/' + mask_r_path.split('/')[-1],''),exist_ok=True)
        os.makedirs(mask_all_path.replace('/' + mask_all_path.split('/')[-1],''),exist_ok=True)
        os.makedirs(label_mask_d_path.replace('/' + label_mask_d_path.split('/')[-1],''),exist_ok=True)
        os.makedirs(label_mask_g_path.replace('/' + mask_g_path.split('/')[-1],''),exist_ok=True)
        os.makedirs(label_mask_r_path.replace('/' + mask_r_path.split('/')[-1],''),exist_ok=True)
        os.makedirs(label_mask_all_path.replace('/' + mask_all_path.split('/')[-1],''),exist_ok=True)
    
        cv2.imwrite(mask_d_path, mask_d_im)
        cv2.imwrite(mask_g_path,  mask_g_im)
        cv2.imwrite(mask_r_path, mask_r_im)
        cv2.imwrite(mask_all_path, mask_all_im)
        cv2.imwrite(label_mask_d_path, label_masked_d_color)
        cv2.imwrite(label_mask_g_path, label_masked_g_color)
        cv2.imwrite(label_mask_r_path, label_masked_r_color)
        cv2.imwrite(label_mask_all_path, label_masked_all_color)

  return label_masked_all


if __name__ == '__main__':

  # data_path = '/media/jiayu/share/win10data/datasets/MS2'
  # scenes = glob.glob(data_path + '/sync_data/*', recursive=False)
  # print(scenes[3])

  # for scene in scenes[3:4]:
  #   print(scene)

  #   label_files = sorted(glob.glob(scene + '/rgb/label_cre/*.npy'))
  #   print(len(label_files))

  #   for label_file in label_files[:100]:
  #     left_file = label_file.replace('label_cre', 'img_left').replace('npy', 'png')
  #     right_file = label_file.replace('label_cre', 'img_right').replace('npy', 'png')
  #     label_filter(left_file, right_file, label_file)

  data_path = '/media/jiayu/9E72-0783/neu-sc2/data/311a/scene0/processed/train/scene0/nir_nir'

  label_files = sorted(glob.glob(data_path + '/label_gt/*.npy'))

  for label_file in label_files[:]:
    left_file = label_file.replace('label_gt', 'left').replace('npy', 'png')
    right_file = label_file.replace('label_gt', 'right').replace('npy', 'png')
    label_filter(left_file, right_file, label_file, show=True)