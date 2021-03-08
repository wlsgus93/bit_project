
import pyrealsense2 as rs
import numpy as np
import cv2

# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path


cfg=rs.config()
cfg.enable_record_to_file("byung_hi.bag")
pipeline = rs.pipeline()
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
pipeline.start(cfg)
colorizer = rs.colorizer()
while True:
        # Get frameset of depth
    frames = pipeline.wait_for_frames()

        # Get depth frame
    depth_frame = frames.get_depth_frame()

        # Colorize depth frame to jet colormap
    depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
    depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
    cv2.imshow("Depth Stream", depth_color_image)
    key = cv2.waitKey(1)
        # if pressed escape exit program
    if key == 27:
        cv2.destroyAllWindows()
        break

