
import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()    # 이미지 가져옴
config = rs.config()        # 설정 파일 생성
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  #크기 , 포맷, 프레임 설정
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)   #설정을 적용하여 이미지 취득 시작, 프로파일 얻음

depth_sensor = profile.get_device().first_depth_sensor()    # 깊이 센서를 얻음
depth_scale = depth_sensor.get_depth_scale()                # 깊이 센서의 깊이 스케일 얻음
print("Depth Scale is: ", depth_scale)

