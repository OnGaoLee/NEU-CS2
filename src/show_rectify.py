import cv2
import sys
import argparse

def showRectify(left, right, output_path):

    rectifyL = cv2.imread(left,cv2.IMREAD_COLOR)
    rectifyR = cv2.imread(right,cv2.IMREAD_COLOR)
    print(rectifyR.shape)
    rectify_size = rectifyL.shape
    rectifyLR = cv2.hconcat([rectifyL, rectifyR])

    cv2.rectangle(rectifyLR, (0,0), (rectify_size[1],rectify_size[0]),(255,0,0),3)
    cv2.rectangle(rectifyLR, (rectify_size[1],0), (2*rectify_size[1], rectify_size[0]),(255,0,0),3)


    h, w, _ = rectifyLR.shape

    scalar = 1200/max(h, w) # 缩放图像宽度为1200，方便可视化
    H = int(h * scalar)
    W = int(w * scalar)
    scaleMixLR = cv2.resize(rectifyLR, (W,H), interpolation=cv2.INTER_LINEAR)

    for i in range(0, H, 16):
        cv2.line(scaleMixLR, (0,i), (W,i), (0,255,0))

    cv2.namedWindow("result",cv2.WINDOW_AUTOSIZE)
    cv2.imshow("result",scaleMixLR)
    key = cv2.waitKey(0)
    if key ==27:
      raise Exception
    if key == ord('s'):
      cv2.imwrite(output_path + "/rectify_result.png",scaleMixLR)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Preprocess MS2 dataset')
  parser.add_argument('--left', type=str, required=True, default="/media/jiayu/82F0-E384/neu-sc2/data/processed/train/311a/nir_rgb/left/scene0_58.png", help='Path to load left img')
  parser.add_argument('--right', type=str, required=True, default="/media/jiayu/82F0-E384/neu-sc2/data/processed/train/311a/nir_rgb/right/scene0_58.png", help='Path to load right img')
  parser.add_argument('--output_path', type=str, required=True, default="/media/jiayu/82F0-E384/neu-sc2/data/processed_rectify", help='Path to save processed MS2 dataset')

  args = parser.parse_args()

  showRectify(args.left,args.right,args.output_path)