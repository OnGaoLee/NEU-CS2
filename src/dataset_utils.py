import math
import cv2
import numpy as np

def get_fov(h, w, focal):

  hfov = 2*math.atan(0.5*h/focal)
  wfov = 2*math.atan(0.5*w/focal)
  return hfov, wfov

def getMappingMatrix(transform_mtx, target):
  '''获得单应性变换映射'''
  mapx = np.zeros_like(target)
  mapy = np.zeros_like(target)
  h,w = target.shape
  for u in range(w):
    for v in range(h):
      u_ = transform_mtx[0,0]*u + transform_mtx[0,1]*v + transform_mtx[0,2]
      v_ = transform_mtx[1,0]*u + transform_mtx[1,1]*v + transform_mtx[1,2]
      c = transform_mtx[2,0]*u + transform_mtx[2,1]*v + transform_mtx[2,2]
      mapx[v,u] = u_/c
      mapy[v,u] = v_/c
  return mapx, mapy

def imgRemap(img_src, mapx, mapy):
    '''imgRemap图像重映射'''
    size_src = np.shape(img_src)
    h_dst,w_dst = np.shape(mapx)
    if len(size_src) == 2:
      img_dst = np.zeros((h_dst, w_dst))
      for i in range(h_dst):
          for j in range(w_dst):
              x = mapx[i, j]
              y = mapy[i, j]
              x_int = math.floor(x)
              y_int = math.floor(y)
              px = x - x_int
              py = y - y_int
              if ((x >= float(0)) & (x < float(size_src[1]-1)) & (y >= float(0)) & (y < float(size_src[0]-1))):
                  # 双线性插值
                  img_dst[i, j] = (1 - px) * (1 - py) * img_src[y_int, x_int] + (1 - py) * px * img_src[y_int, x_int + 1] + py * (1 - px) * img_src[y_int + 1, x_int] + px * py * img_src[y_int + 1, x_int + 1]
              else:
                  img_dst[i, j] = 0
    if len(size_src) == 3:
      img_dst = np.zeros((h_dst, w_dst, 3))
      for i in range(h_dst):
          for j in range(w_dst):
              x = mapx[i, j]
              y = mapy[i, j]
              x_int = math.floor(x)
              y_int = math.floor(y)
              px = x - x_int
              py = y - y_int
              if ((x >= float(0)) & (x < float(size_src[1]-1)) & (y >= float(0)) & (y < float(size_src[0]-1))):
                  # 双线性插值
                  img_dst[i, j, 0] = (1 - px) * (1 - py) * img_src[y_int, x_int, 0] + (1 - py) * px * img_src[y_int, x_int + 1, 0] + py * (1 - px) * img_src[y_int + 1, x_int, 0] + px * py * img_src[y_int + 1, x_int + 1, 0]
                  img_dst[i, j, 1] = (1 - px) * (1 - py) * img_src[y_int, x_int, 1] + (1 - py) * px * img_src[y_int, x_int + 1, 1] + py * (1 - px) * img_src[y_int + 1, x_int, 1] + px * py * img_src[y_int + 1, x_int + 1, 1]
                  img_dst[i, j, 2] = (1 - px) * (1 - py) * img_src[y_int, x_int, 2] + (1 - py) * px * img_src[y_int, x_int + 1, 2] + py * (1 - px) * img_src[y_int + 1, x_int, 2] + px * py * img_src[y_int + 1, x_int + 1, 2]
              else:
                  img_dst[i, j, :] = 0
    return img_dst 

def apply_forward_homography_transform(input_image, H, output_image_size):
  """自定义正向映射"""
  height, width = input_image.shape
  output_width, output_height = output_image_size
  output_image = np.zeros((output_height, output_width))

  for y in range(height):
      for x in range(width):
          p = np.array([x, y, 1])
          p_transformed = np.dot(H, p)
          p_transformed = p_transformed / p_transformed[2]
          x_new, y_new, _ = p_transformed
          if (x_new >=0) & (y_new >=0) & (x_new < (output_width-1)) & (y_new < (output_height-1)):

            x_floor, y_floor = int(np.floor(x_new)), int(np.floor(y_new))
            # x_ceil, y_ceil = int(np.ceil(x_new)), int(np.ceil(y_new))

            dist = []

            for i in [0, 1]:
                for j in [0, 1]:
                    x_neighbor = x_floor + i
                    y_neighbor = y_floor + j
                    dist.append(np.sqrt((x_new - x_neighbor)**2 + (y_new - y_neighbor)**2))

            min_dist_idx = np.argmin(dist)
            # print(min_dist_idx)
            nearest_x = x_floor + np.uint8(min_dist_idx >=2)
            nearest_y = y_floor + min_dist_idx % 2

            output_image[nearest_y, nearest_x] = input_image[y, x]

  return output_image

def disp_transform(disp_src, f_src, baseline_src, f_dst, baseline_dst, size_dst, depth_mtx, homo_mtx, depth=False, forward_mapping=False):
  '''不同立体对之间的视差变换'''
  size_src = np.shape(disp_src)
  if not depth:
    depth_src = f_src * baseline_src / (disp_src + 1e-6)
  else:
    depth_src = disp_src

  mtx1 = depth_mtx[2,0]*np.ones_like(disp_src)
  mtx2 = depth_mtx[2,1]*np.ones_like(disp_src)
  mtx3 = depth_mtx[2,2]*np.ones_like(disp_src)

  u_array = np.repeat([np.array(range(size_src[1]))],size_src[0],axis=0)
  v_array = np.repeat(np.array([range(size_src[0])]).T,size_src[1],axis=1)

  depth_src_dst = depth_src * (mtx1 * u_array + mtx2 * v_array + mtx3)
  if not forward_mapping:
    depth_dst = cv2.warpPerspective(depth_src_dst, homo_mtx, (size_dst[1], size_dst[0]))
  else:
    depth_dst = apply_forward_homography_transform(depth_src_dst, homo_mtx, (size_dst[1], size_dst[0]))
    
  mask1 = 1- np.float32(depth_dst <= 0)

  disp_dst = mask1 * f_dst * baseline_dst / (depth_dst + 1e-6)
  mask2 = 1 - np.float32(disp_dst > 192)
  mask3 = 1 - np.float32(disp_dst < 0.1)

  return disp_dst * mask2 * mask3


