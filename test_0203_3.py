import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import pyzbar.pyzbar as pyzbar
blk_size=9
C=5
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgra8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Convert images RGB to Gray Scale
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY) #3ch -> 1ch
        _,gray=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #2021_0204_add code
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
        #                             cv2.THRESH_BINARY, blk_size, C)


        decoded = pyzbar.decode(color_image)

        for d in decoded:
            x, y, w, h = d.rect
            barcode_data = d.data.decode("utf-8")
            barcode_type = d.type

            cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2) #2021_0204 kjh add
            text = '%s (%s)' % (barcode_data, barcode_type)
            print(text)
            cv2.putText(color_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(gray, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA) #2021_0204 kjh add
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
       # images = np.hstack((color_image, depth_colormap))
        #images=np.hstack((color_image,color_image))
        # Show images
        cv2.imwrite('img1/' + color_path, color_image)
        cv2.imshow('gray',gray) #2021_0204 kjh add
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense',color_image)

        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
