import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import pyzbar.pyzbar as pyzbar
import io
# Configure depth and color streams
f= open('./barcode.txt','w')
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60) # Depth stream info
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60) # RGB stream info
x1=int(640/3)
y1=int(480/2)

# Start streaming
pipeline.start(config)
i=0
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

        # images ROI
        roi = cv2.rectangle(color_image, (x1, y1), (x1 + x1, y1 + x1), (0, 255, 0), 2)
        # roi=color_image[x:2*x,y:2*y]
        # img2=roi.copy()


        decoded = pyzbar.decode(gray)

        for d in decoded:

            x, y, w, h = d.rect # 인식좌표 포인트
            if x>=x1 and y>=y1 and x+w<=2*x1 and y+h<=2*y1: # 인식범위
                barcode_data = d.data.decode("utf-8")
                barcode_type = d.type

            # cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2) #2021_0204 kjh add
                text = '%s (%s)' % (barcode_data, barcode_type)

                print(text+'{}'.format(i))
                cv2.putText(color_image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(gray, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA) #2021_0204 kjh add
                i += 1
                if barcode_data is not None:
                    f.write(barcode_data)
                    break



        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
       # images = np.hstack((color_image, depth_colormap))
        #images=np.hstack((color_image,color_image))
        # Show images
        #cv2.imshow('gray',gray) #2021_0204 kjh add

        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense',color_image)
        cv2.waitKey(1)

finally:
    # Stop streaming
    pipeline.stop()
