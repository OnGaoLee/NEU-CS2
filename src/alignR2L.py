import cv2
import numpy as np
import torch

def align(left_path, right_path, disp_path):
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    disp = np.load(disp_path)
    left_t = torch.Tensor(left).unsqueeze(0).permute(0,3,1,2)
    right_t = torch.Tensor(right).unsqueeze(0).permute(0,3,1,2)
    disp_t = torch.Tensor(disp).unsqueeze(0).unsqueeze(0)

    h, w = right.shape[-2:]
    ys, xs = torch.meshgrid([torch.arange(h), torch.arange(w)])
    id_coords = torch.stack((xs, ys),dim=2).float()
    theta = torch.Tensor(np.array([[1,0,0],[0,1,0]])).expand(disp_t.shape[0],2,3).contiguous()
    # print(disp_t)
    grid = torch.nn.functional.affine_grid(theta, disp_t.size())
    disp_t = disp_t.permute(0,2,3,1)
    disp_t = torch.cat((disp_t,torch.zeros_like(disp_t)),3)
    # print(disp_t.shape[2])
    grid = grid - 2*disp_t/disp_t.shape[2]
    sampled = torch.nn.functional.grid_sample(right_t.float(), grid.float())
    # print(right_t.shape)
    # print(sampled.size())
    return sampled.squeeze(0).permute(1,2,0).cpu().numpy() 

if __name__ == '__main__':
    left_path = '/media/jiayu/82F0-E384/neu-sc2/data/processed/train/outdoor/nir_rgb/left/scene0_9.png'
    right_path = left_path.replace('left','right')
    disp_path = left_path.replace('left','label_cre_unfiltered').replace('png','npy')

    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    left_warp = align(left_path,right_path,disp_path)
    origin = np.vstack([left,right])
    compare = np.vstack([left, np.uint8(left_warp)])
    h, w, _ = compare.shape
    for i in range(0, w, 16):
        cv2.line(compare, (i,0), (i,h), (0,255,0))
    for i in range(0, w, 16):
        cv2.line(origin, (i,0), (i,h), (0,255,0))
    all = np.hstack([origin,compare])
    cv2.imshow('all', all)
    key =cv2.waitKey(0)
    if key == ord('s'):
      cv2.imwrite('./align_example8.png',all)