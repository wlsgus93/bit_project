import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import matplotlib.pyplot

NowDate = datetime.datetime.now()



pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

color_path = NowDate.strftime('%Y-%m-%d_%H%M_'+'.png')
depth_path = NowDate.strftime('%Y-%m-%d_%H%M_'+'Depth.png')
# colorwriter = cv2.VideoWriter(color_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)
# depthwriter = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480), 1)


profile=pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
up_x=296
up_y=240

down_x=296
down_y=320
try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # print(depth_frame.get_frame_metadata())
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        #print(depth_frame.get_data)
        dist_to_center_up = depth_frame.get_distance(up_y, up_x)
        dist_to_center_down = depth_frame.get_distance(down_y,down_x)
        up_text='The camera is facing an object:'+ str(dist_to_center_up)+ 'meters away'
        down_text='The camera is facing an object:' +str(dist_to_center_down)+ 'meters away'
        #print('The camera is facing an object:', dist_to_center, 'meters away')

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        #print(type(depth_scale))
        print(str(depth_scale*100)+'cm')
        #convert images to numpy arrays

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # colorwriter.write(color_image)
        # depthwriter.write(depth_colormap)
        cv2.circle(depth_colormap,(up_x,up_y),3,(0,0,255))
        cv2.circle(depth_colormap, (down_x, down_y), 3, (0, 0, 255))
        cv2.putText(depth_colormap, up_text,(up_x, up_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_colormap, down_text,(down_x, down_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Stream', depth_colormap)
        cv2.imshow('color map',color_image)
        if cv2.waitKey(1) == 65:
            cv2.imwrite('img/'+color_path,color_image)
        if cv2.waitKey(1) == ord("q"):
            cv2.imwrite('img/' + color_path , color_image)
            break

finally:
    # colorwriter.release()
    # depthwriter.release()
    pipeline.stop()



