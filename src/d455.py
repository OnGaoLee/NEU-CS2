# -*- coding:utf-8 -*-
"""
@FileName  :D455.py
@Time      :2024/4/15 17:52
@Author    :Jiayu

"""
import numpy as np
import pyrealsense2 as rs
import cv2
import os
import time

# 设定分辨率等相机参数
width = 1280
height = 720
fps = 30
device_id = "135122250814"  # 是D455的序列号（可以在相机上找到），可以设置为None，
# device_id = None
# 记录拍摄图像数
index = 521
enable_depth = False
# enable_depth = True
# TODO: enable_rgb = True
# TODO: enable_ir = True
if __name__ == '__main__':
# 1.初始化
    '============================================================================================'
    # 开始准备相机
    pipeline = rs.pipeline()
    # 属性设置
    config = rs.config()
    # 启动指定设备
    if None != device_id:
        config.enable_device(device_id)
    # 深度图
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    # IR左图
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)  # 默认格式 rs.format.y8
    # IR右图
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)  # 默认格式 rs.format.y8
    # 彩色图
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    # 启用相机
    profile = pipeline.start(config)
    intrinsics_depth = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    print(intrinsics_depth.coeffs) 
    intrinsics_rgb = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    print(intrinsics_rgb) 
    intrinsics_infrared1 = profile.get_stream(rs.stream.infrared,1).as_video_stream_profile().get_intrinsics()
    print(intrinsics_infrared1) 
    extrinsics_rgb_infrared1 = profile.get_stream(rs.stream.infrared,1).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.color))
    print(extrinsics_rgb_infrared1) 
    intrinsics_infrared2 = profile.get_stream(rs.stream.infrared,2).as_video_stream_profile().get_intrinsics()
    print(intrinsics_infrared2) 
    extrinsics_infrared1_infrared2 = profile.get_stream(rs.stream.infrared,1).as_video_stream_profile().get_extrinsics_to(profile.get_stream(rs.stream.infrared,2))
    print(extrinsics_infrared1_infrared2) 

    # 保存相机参数
    nir_intrinsics_1 = np.array([[intrinsics_infrared1.fx,0,intrinsics_infrared1.ppx],[0,intrinsics_infrared1.fy,intrinsics_infrared1.ppy],[0,0,1]])
    nir_coeffs_1 = np.array(intrinsics_infrared1.coeffs)
    nir_height_1 = intrinsics_infrared1.height
    nir_width_1 = intrinsics_infrared1.width

    nir_intrinsics_2 = np.array([[intrinsics_infrared2.fx,0,intrinsics_infrared2.ppx],[0,intrinsics_infrared2.fy,intrinsics_infrared2.ppy],[0,0,1]])
    nir_coeffs_2 = np.array(intrinsics_infrared2.coeffs)
    nir_height_2 = intrinsics_infrared2.height
    nir_width_2 = intrinsics_infrared2.width

    R_infrared1_infrared2 = np.array([np.array(extrinsics_infrared1_infrared2.rotation[:3]),np.array(extrinsics_infrared1_infrared2.rotation[3:6]),np.array(extrinsics_infrared1_infrared2.rotation[6:])])
    T_infrared1_infrared2 = np.array(extrinsics_infrared1_infrared2.translation)

    rgb_intrinsics = np.array([[intrinsics_rgb.fx,0,intrinsics_rgb.ppx],[0,intrinsics_rgb.fy,intrinsics_rgb.ppy],[0,0,1]])
    rgb_coeffs = np.array(intrinsics_rgb.coeffs)
    rgb_height = intrinsics_rgb.height
    rgb_width = intrinsics_rgb.width

    R_infrared1_rgb = np.array([np.array(extrinsics_rgb_infrared1.rotation[:3]),np.array(extrinsics_rgb_infrared1.rotation[3:6]),np.array(extrinsics_rgb_infrared1.rotation[6:])])
    T_infrared1_rgb = np.array(extrinsics_rgb_infrared1.translation)

    calib_ori = cv2.FileStorage('./calib_ori.xml', cv2.FILE_STORAGE_WRITE)

    calib_ori.write('K_nirL', nir_intrinsics_1)
    calib_ori.write('D_nirL', nir_coeffs_1)
    calib_ori.write('H_nirL', nir_height_1)
    calib_ori.write('W_nirL', nir_width_1)

    calib_ori.write('K_nirR', nir_intrinsics_2)
    calib_ori.write('D_nirR', nir_coeffs_2)
    calib_ori.write('H_nirR', nir_height_2)
    calib_ori.write('W_nirR', nir_width_2)

    calib_ori.write('R_nir2nir', R_infrared1_infrared2)
    calib_ori.write('T_nir2nir', T_infrared1_infrared2)

    calib_ori.write('K_rgbR', rgb_intrinsics)
    calib_ori.write('D_rgbR', rgb_coeffs)
    calib_ori.write('H_rgbR', rgb_height)
    calib_ori.write('W_rgbR', rgb_width)

    calib_ori.write('R_nir2rgb', R_infrared1_rgb)
    calib_ori.write('T_nir2rgb', T_infrared1_rgb)

    calib_ori.release()

    sensor = pipeline.get_active_profile().get_device().query_sensors()[0]
    # 关闭/打开激光发射器  0 / 1
    if enable_depth:
      sensor.set_option(rs.option.emitter_enabled, 1)
    else:
      sensor.set_option(rs.option.emitter_enabled, 0)
    # 设置相机的增益      min-16  max-248  default-1  step-16
    sensor.set_option(rs.option.gain, 32)
    # 设置相机的曝光      min-1    max-165000    default-1    step-33000
    sensor.set_option(rs.option.exposure, 70000)
    # 设置自动曝光        0 / 1
    sensor.set_option(rs.option.enable_auto_exposure, 1)

    sensor_rgb = pipeline.get_active_profile().get_device().query_sensors()[1]
    # 设置相机的增益      min-16  max-248  default-1  step-16
    sensor_rgb.set_option(rs.option.gain, 32)
    # 设置相机的曝光      min-1    max-165000    default-1    step-33000
    sensor_rgb.set_option(rs.option.exposure, 10000)
    # 设置自动曝光        0 / 1
    sensor_rgb.set_option(rs.option.enable_auto_exposure, 1)
    # 设置自动白平衡        0 / 1
    sensor_rgb.set_option(rs.option.enable_auto_white_balance, 1)

    # 记录帧数
    frame_count = 0
    # 记录开始时间
    start_time = time.time()
    frame_time = start_time
    # 获得深度传感器的深度缩放因子 0.0010000000474974513 ~ 0.001
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # 创建窗口，用来显示读取到的每一帧图像
    cv2.namedWindow("all", cv2.WINDOW_NORMAL)

    # 创建保存视频的对象，设置编码格式，帧率，图像的宽高等
    fourcc = 0x00000002  # 设置编码格式，一般情况下fourcc = cv2.VideoWriter_fourcc(*'FLV1')   *'FLV1'表示编码方式
    # cv2.VideoWriter_fourcc()返回一个16进制数
    # 示例：out = cv2.VideoWriter('color.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))
    # outVedio_RGB = cv2.VideoWriter('./RGB.flv', fourcc, fps, (width, height))
    # outVedio_IR_Left = cv2.VideoWriter('./IR_Left.flv', fourcc, fps, (width, height))
    # outVedio_IR_Right = cv2.VideoWriter('./IR_Right.flv', fourcc, fps, (width, height))
    # outVedio_Depth = cv2.VideoWriter('./Depth.flv', fourcc, fps, (width, height))
    # outVedio_RGB = cv2.VideoWriter('./RGB.avi', cv2.VideoWriter_fourcc(*'RGBA'), fps//2, (width, height))
    # outVedio_IR_Left = cv2.VideoWriter('./IR_Left.avi', cv2.VideoWriter_fourcc(*'RGBA'), fps//2, (width, height))
    # outVedio_IR_Right = cv2.VideoWriter('./IR_Right.avi', cv2.VideoWriter_fourcc(*'RGBA'), fps//2, (width, height))
    '''
    * cv2.VideoWriter参数:
        1-需保存视频的文件名
        2-解码参数  0x00000002/cv2.VideoWriter_fourcc(*'FLV1')/cv2.VideoWriter_fourcc('F','L','V','1')都表示Flash视频，如果想用h264编码则可以用0x00000007，不过这需要安装Openh264的库
        3-fps帧率
        4-分辨率
        5-True为彩色视频, False为黑白视频
    * cv2.VideoWriter_fourcc('R', 'G', 'B', 'A') 未压缩，但会产生大文件。文件扩展名应为.avi。
    * cv2.VideoWriter_fourcc('I', '4', '2', '0') 这个选项是一个未压缩的YUV编码，4: 2:0色度子采样。这种编码广泛兼容，但会产生大文件。文件扩展名应为.avi。
    * cv2.VideoWriter_fourcc('P','I','M','1') 此选项为MPEG-1。文件扩展名应为.avi。
    * cv2.VideoWriter_fourcc('X','V','I','D') 此选项是一个相对较旧的MPEG-4编码。视频大小为平均值，MPEG4所需要的空间是MPEG1或M-JPEG的1/10，它对运动物体可以保证有良好的清晰度，时间/画质具有可调性如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.avi。
    * cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 此选项是另一个相对较旧的MPEG-4编码。如果要限制结果视频的大小，这是一个很好的选择。文件扩展名应为.m4v。
    * cv2.VideoWriter_fourcc('X','2','6','4')  这个选项是一种比较新的MPEG-4编码方式。如果你想限制结果视频的大小，这可能是最好的选择。文件扩展名应为.mp4。
    * cv2.VideoWriter_fourcc('T','H','E','O') 这个选项是Ogg Vorbis。文件扩展名应为.ogv。
    * cv2.VideoWriter_fourcc('F','L','V','1') 此选项为Flash视频。FLV流媒体格式是一种新的视频格式。由于它形成的文件极小、加载速度极快，使得网络观看视频文件成为可能，文件扩展名应为.flv。
    '''

    while True:

# 2.获取数据流
        '============================================================================================'
        # 等待图像帧
        frameset = pipeline.wait_for_frames()
        # 创建对齐
        # 获取深度图
        frame_depth = frameset.get_depth_frame()  # 没有和RGB对齐的深度图
        frame_depth = rs.decimation_filter(1).process(frame_depth) # 中值滤波
        frame_depth = rs.disparity_transform(True).process(frame_depth) # 视差表示
        frame_depth = rs.spatial_filter().process(frame_depth) # 空间滤波
        frame_depth = rs.disparity_transform(False).process(frame_depth) # 深度表示
        # 获取IR左右图像对
        frame_left = frameset.get_infrared_frame(1)
        frame_right = frameset.get_infrared_frame(2)
        # 获取RGB图像
        frame_color = frameset.get_color_frame()  # 没有和深度图对齐的RGB图

# 3.格式转换
        '============================================================================================'
        # 转换图像帧格式，以便后面的显示和储存
        depth_image = np.asanyarray(frame_depth.get_data())
        left_image = np.asanyarray(frame_left.get_data())
        right_image = np.asanyarray(frame_right.get_data())
        RGB_image = np.asanyarray(frame_color.get_data())

        # 将Y8格式转换为uint8并扩充为RGB，这样做为了能够保存IR的视频
        left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
        # 给深度图上伪彩色
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        '''
        dst = cv2.convertScaleAbs(src , alpha=0.03 , beta);  
        假设我们需要让我们的深度摄像头感兴趣的距离范围有差别地显示，那么我们就需要确定一个合适的alpha值，
        公式为：有效距离*alpha=255，假设我们想让深度摄像头8m距离内的深度被显示，>8m的与8m的颜色显示相同，
        那么alpha=255/(8*10^3)≈0.03，假设我们想让深度摄像头6m距离内的深度被显示，>6m的与6m的颜色显示相同，
        那么alpha=255/(6*10^3)≈0.0425，cv.convertScaleAbs()函数对输入数组进行如下运算：
        dst = 截取8位(|src*alpha+beta|)

        '''
        # 水平拼接图像对
        image_all = np.hstack((left_image, right_image, RGB_image,  depth_colormap))  

# 4.显示与保存
        '============================================================================================'
        # 显示图像
        cv2.imshow("all", image_all)

        # 保存视频
        # 将每一帧图像写入到输出文件中
        # outVedio_RGB.write(RGB_image)  # 原RGB视频写入
        # outVedio_IR_Left.write(left_image)  # IR_Left视频写入
        # outVedio_IR_Right.write(right_image)  # IR_Right视频写入

# 5.退出
        '============================================================================================'
        frame_count += 1
        key = cv2.waitKey(1)
        if (key & 0xFF == ord('q')) | (key & 0xFF == 27):
            pipeline.stop()
            cv2.destroyAllWindows()
            # 释放资源
            # outVedio_RGB.release()
            # outVedio_IR_Left.release()
            # outVedio_IR_Right.release()
            cv2.destroyAllWindows()
            # 记录停止时间
            total_time = time.time() - start_time
            print(total_time)
            break
        elif key & 0xFF == ord('s'):
            
            if enable_depth:
                os.makedirs("./depth",exist_ok=True)
                cv2.imwrite("./depth/{}.png".format(index), depth_colormap)
                depth_real = depth_image * depth_scale
                np.save("./depth/{}.npy".format(index), depth_real)
            else:
                os.makedirs("./nir_nir/left",exist_ok=True)
                cv2.imwrite("./nir_nir/left/{}.png".format(index), left_image)
                os.makedirs("./nir_nir/right",exist_ok=True)
                cv2.imwrite("./nir_nir/right/{}.png".format(index), right_image)
                os.makedirs("./rgb",exist_ok=True)
                cv2.imwrite("./rgb/{}.png".format(index), RGB_image)
                if not os.path.exists("./depth/{}.npy".format(index)):
                    os.makedirs("./depth",exist_ok=True)
                    cv2.imwrite("./depth/{}.png".format(index), depth_colormap)
                    depth_real = depth_image * depth_scale
                    np.save("./depth/{}.npy".format(index), depth_real)
            index += 1
