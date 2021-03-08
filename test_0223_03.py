import pyrealsense2 as rs
import numpy as np
import cv2


config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

config.enable_device_from_file("./byung_hi.bag")
pipeline = rs.pipeline()
pipeline.start(config)
colorizer = rs.colorizer()
i=0
f=open('/home/user/librealsense/ex_test/new.txt','w')
try:
    while True:

        frames = pipeline.wait_for_frames()

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

# Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        # cv2.imwrite('/home/user/librealsense/ex_test' + "/" + str(i).zfill(6) + ".png", depth_image)
        print(depth_frame.get_data())
        print(str(depth_image))

        i+=1
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)



        key = cv2.waitKey(1)
        if key == 81:
            pass
        if key == 27:
            f.write(str(depth_image))
            f.close()
            cv2.destroyAllWindows()
            break

finally:

    # Stop streaming
    pipeline.stop()

