
import pyrealsense2 as rs
import numpy as np
import cv2
import PIL
from PIL import Image
from time import sleep
video_strime=True
# 1. set max or min distance
# 2. hole filter
# 3. allign depth and color
# 4. delete bg
# 5. image processing
# 5.1 convert color, ad_threshhold ,
# 5.2 blur(gaussian or bilateral) and edge detection
# 5.3 edge detection (sobel,sharr,raplacian,canny)
#


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
#0~~65000
# Start streaming
profile= pipeline.start(config)



# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

#value setting
max_distance = 1.5
object_min_distance = 0.5
object_width=0
object_height=0
object_depth=0.18
clipping_distance_in_meters = max_distance # unit : meter
clipping_distance_in_meters_2= object_min_distance
clipping_distance = clipping_distance_in_meters / depth_scale
clipping_distance_2=clipping_distance_in_meters_2/depth_scale
blk_size=9
C=5
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0
kernel_3=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
kernel_11 = np.ones((5, 5), np.uint8)
print(clipping_distance)
print(clipping_distance_2)
print(depth_scale)

#depth sensor setup
hole_filling = rs.hole_filling_filter()
threshold_filling=rs.threshold_filter(1,16)
colorizer = rs.colorizer()
colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
colorizer.set_option(rs.option.min_distance, object_min_distance)
colorizer.set_option(rs.option.max_distance, max_distance)

# print(kernel_3)
#aligned

align_to = rs.stream.color
align = rs.align(align_to)
i=0
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        color_frame = frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        filtered = hole_filling.process(aligned_depth_frame)
        # filtered = threshold_filling.process(filtered)
        if not aligned_depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(filtered.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # print(depth_image)
        depth_colormap = np.asanyarray(colorizer.colorize(filtered).get_data())
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
        depth_image_3d_1=np.asanyarray(colorizer.colorize(filtered).get_data())
        cv2.imshow('depth_image',depth_image)
        # cv2.imshow('depth_image_3d_1',depth_image_3d_1)
        if i%5==0:
            sleep(0.4)
        cv2.circle(depth_image_3d_1, (433, 241), 3, (0, 0, 255))
        print((depth_image_3d_1[433:434, 241:242, :]))
        # print(depth_image_3d[433:434,241:242,:])
        print(depth_image[433:434,241:242])
        print('depth_scale :       ',str(depth_scale))
        print(depth_image.dtype)
        asdf=aligned_depth_frame.get_distance(433, 241)
        print(asdf)
        sobelx=cv2.Sobel(depth_colormap,-1,1,0,ksize=3)
        sobely=cv2.Sobel(depth_colormap,-1,0,1,ksize=3)
        cv2.imshow('sobelx',sobelx)
        cv2.imshow('sobely',sobely)
        sobol_m=np.vstack((sobelx,sobely))
        cv2.imshow('merged',sobelx+sobely)
        cv2.imshow('depth_image_3d_1',depth_image_3d_1)
        print(depth_image_3d[247:248,247:248,:])
        i+=1

        # cv2.imshow('depth_colormap',depth_colormap)

        #Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap_apply = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= clipping_distance_2), 0,
                              color_image)
        # cv2.imshow('depth_colormap_apply',depth_colormap_apply)
        # cv2.imshow('bg_removed',bg_removed)
        # print('depth image:' + str(depth_image.shape))
        # print('depth_3d image:' + str(depth_image_3d.shape))
        # print(bg_removed.shape)
        # bgr_gray=cv2.cvtColor(bg_removed,cv2.COLOR_BGR2GRAY)
        depth_image_show=False
        depth_colormap_show=False
################image processing
        # cv2.imshow('bg_removed',bg_removed)
        #make mask Bitwise_and
        mask_img = np.zeros_like(color_image)
        mask_img = cv2.rectangle(mask_img,(170,100),(700,500),(255,255,255),-1) #same height between object and camera
        # cv2.imshow('mask',mask_img)
        dist_to_center_down = aligned_depth_frame.get_distance(437,232)
        # print(dist_to_center_down)
        bg_removed=cv2.bitwise_and(bg_removed, mask_img)
        color_image = cv2.bitwise_and(color_image, mask_img)
        depth_colormap = cv2.bitwise_and(depth_colormap, mask_img)


#convert to gray
        bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
        color_gray=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)

        blur=cv2.bilateralFilter(color_gray,10,10,10)
        blur_depth=cv2.bilateralFilter(bg_removed,10,10,10)
        # print('color:'+str(type(color_image))+str(color_image.shape))
        # print('mask:' + str(type(mask_img))+str(mask_img.shape))



        #erode=cv2.erode(blur,kernel_3,iterations = 1)#src, filter_d,sigmacolor,sigmaspace
        #sharp=cv2.filter2D(bg_removed, -1, kernel_sharpen_3)
        th2= cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                  cv2.THRESH_BINARY,blk_size,C)
        th2_depth = cv2.adaptiveThreshold(blur_depth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, blk_size, C)
        th2_comp=np.hstack((th2,th2_depth))


        edges=cv2.Canny(th2,200,240)
        edges_depth=cv2.Canny(th2_depth,200,240)
        edges_comp = np.hstack((edges, edges_depth))

        result = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_11)
        cv2.imshow('mop',result)
        ###contour

        contours, hierarchy = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        NumeroContornos = str(len(contours))
        total=0
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # print(len(approx))

            # cv2.drawContours(color_image, [approx], 0, (0, 255, 255), 1)

            x, y, w, h = cv2.boundingRect(cnt)
            if w > 100 and h > 40 and 200>w and 100>h :
                x_center = int(x + w / 2)
                y_center = int(y + h / 2)
                dist_to_center = aligned_depth_frame.get_distance(x_center, y_center)
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(color_image, (x, y), 3, (0, 0, 255))
                if dist_to_center > 1 and dist_to_center<1.15:
                    total+=2
                elif dist_to_center >1.20 and dist_to_center<1.4:
                    total+=1

                # print('x,y:{0},{1}'.format(x,y))

                # cv2.circle(color_image, (x,y), 3, (0, 0, 255))

                # print(dist_to_center)

        print('object 갯수:'+ str(total))

        contours_d, hierarchy_d = cv2.findContours(th2_depth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print('asdfasdf'+str(hierarchy))
        # print('asdfasdf'+str(hierarchy_d))
        for cnt in contours_d:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # print(len(approx))

            # cv2.drawContours(depth_colormap, [approx], 0, (0, 0, 255), 1)
            x, y, w, h = cv2.boundingRect(cnt) #output x,y,w,h
            # if w>100 and h>40:
            #     cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), (0, 255, 0), 2)





        #conerharis

        coners=cv2.goodFeaturesToTrack(edges,80,0.01,10)

        #print(type(coners))
        if isinstance(coners, np.ndarray):
            coners = np.int32(coners)
        if isinstance(coners,np.ndarray):

            for coner in coners:
                x,y=coner[0]
                cv2.circle(edges,(x,y),5,(0,0,255),1,cv2.LINE_AA)

        #edges1 = cv2.Canny(th2, 100, 200)
        # lines=cv2.HoughLines(edges,1,np.pi/180,150)
        # lines_depth=cv2.HoughLines(edges_depth,1,np.pi/180,150)
        #
        # # print(isinstance(lines,np.ndarray))
        # find_line=bg_removed.copy()
        #
        # h,w=find_line.shape[:2]
        # if isinstance(lines,np.ndarray):
        #     # print(len(lines))
        #     # print(lines.shape)
        #     for line in lines:
        #         r, theta = line[0]
        #         tx, ty = np.cos(theta), np.sin(theta)
        #         x0, y0 = tx * r, ty * r
        #
        #         x1, y1 = int(x0 + w * (-ty)), int(y0 + h * tx)
        #         x2, y2 = int(x0 - w * (-ty)), int(y0 - h * tx)
        #         # cv2.line(find_line, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #         cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        # if isinstance(lines_depth,np.ndarray):
        #     # print(len(lines))
        #     # print(lines.shape)
        #     for line in lines_depth:
        #         r, theta = line[0]
        #         tx, ty = np.cos(theta), np.sin(theta)
        #         x0, y0 = tx * r, ty * r
        #
        #         x1, y1 = int(x0 + w * (-ty)), int(y0 + h * tx)
        #         x2, y2 = int(x0 - w * (-ty)), int(y0 - h * tx)
        #         # cv2.line(find_line, (x1, y1), (x2, y2), (0, 0, 255), 1)
        #         cv2.line(depth_colormap, (x1, y1), (x2, y2), (0, 0, 255), 1)
        # # conerharis

        # coners = cv2.goodFeaturesToTrack(edges, 80, 0.01, 10)
        # if isinstance(coners, np.ndarray):
        #     coners = np.int32(coners)
        #     for coner in coners:
        #         x, y = coner[0]
        #         cv2.circle(color_image, (x, y), 5, (0, 0, 255), 1, cv2.LINE_AA)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))


        # Show images
        if video_strime:
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)


            cv2.imshow('RealSense', images)
            #cv2.imshow('filtered',th2)
            # cv2.imshow('th2',th2_comp)
            if depth_colormap_show and depth_image_show:
                cv2.imshow('depth_image',depth_image)
                cv2.imshow('depth_3d',depth_image_3d)
            #cv2.imshow('edges1',edges1)

            # cv2.imshow('edges', edges_comp)
            # cv2.imshow('find_line',find_line)
            # print(type(filtered)) fliterd <class:frame>
            cv2.waitKey(1)
        # cv2.waitKey(1)




#  image, light
finally:

    # Stop streaming
    pipeline.stop()