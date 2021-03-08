## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# colorizer = rs.colorizer()
# colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
# colorizer.set_option(rs.option.min_distance, value_min)
# colorizer.set_option(rs.option.max_distance, value_max)
# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)


# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 0.5 #1 meter
clipping_distance_in_meters_2=0.4
clipping_distance = clipping_distance_in_meters / depth_scale
clipping_distance_2=clipping_distance_in_meters_2/depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)
#setup

hole_filling = rs.hole_filling_filter()
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        filtered = hole_filling.process(aligned_depth_frame)
        #depth = aligned_depth_frame.get_depth_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        # print(type(aligned_depth_frame))

        #mask
        depth_image = np.asanyarray(filtered.get_data())
        color_image = np.asanyarray(color_frame.get_data())



        # if aligned_depth_frame(30):
        #     print(type(depth_image))
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        #print(depth_image_3d)
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= clipping_distance_2), 0, color_image)
        #print(clipping_distance)
        #print(depth_image_3d.shape)

        #print(depth_image.shape)
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #print(depth_colormap.shape)
        # depth_colormap[:,:,0]=0
        #
        # depth_colormap[:,:,2]=0
        #
        # print(aligned_depth_frame.get_distance(323,323))
        # for y in range(480):
        #     for x in range(640):
        #         dist1 = (aligned_depth_frame.get_distance(x, y))*1000 #mm unit
        #         dist=aligned_depth_frame.get_distance(x, y)
        #         if 0 < dist and dist < 1:
        #
        #             depth_colormap[:,:,1]=dist1/256



        depth_image = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', images)
        cv2.imshow('color',color_image[:,:,1])
        cv2.imshow('depth_image',depth_image_3d)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
