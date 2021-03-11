
import pyrealsense2 as rs
import numpy as np
import cv2
from time import sleep
import datetime

#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
video_strime=True

NowDate = datetime.datetime.now()
color_path = NowDate.strftime('%Y-%m-%d_%H%M_'+'.png')
# 1. set max or min distance
# 2. hole filter
# 3. allign depth and color
# 4. delete bg
# 5. image processing
# 5.1 convert color, ad_threshhold ,
# 5.2 blur(bilateral) and edge detection
# 5.3 edge detection (mopology,canny)
#


def crop(color_image,depth_colormap,bg_removed,box_model):
    pass


def mask(color_image,depth_colormap): #input image
    #cv2.imshow('mask',mask)
    ########################################make mask Bitwise_and
    mask_img = np.zeros_like(color_image)
    mask_img = cv2.rectangle(mask_img, (170, 0), (750, 700), (255, 255, 255), -1)  # same height between object and camera
    # cv2.imshow('mask',mask_img)
    # bg_removed = cv2.bitwise_and(bg_removed, mask_img)
    color_image = cv2.bitwise_and(color_image, mask_img)
    depth_colormap = cv2.bitwise_and(depth_colormap, mask_img)
    return color_image,depth_colormap
def blur(color_image,depth_colormap):
    blur = cv2.bilateralFilter(color_gray, 10, 10, 10)
    blur_depth = cv2.bilateralFilter(bg_removed, 10, 10, 10)
    cv2.imshow('2', blur)
    return color_image, blur_depth

# def
def get_distance(x,y):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img2 = clahe.apply(img)
    shelf_depth = 0.45
    max_distance = 1 + shelf_depth
    object_min_distance = 0.7
    # object_width = width
    # object_height = height
    # object_depth = depth
    clipping_distance_in_meters = max_distance + shelf_depth  # unit : meter
    clipping_distance_in_meters_2 = object_min_distance
    clipping_distance = clipping_distance_in_meters / depth_scale
    clipping_distance_2 = clipping_distance_in_meters_2 / depth_scale
    hole_filling = rs.hole_filling_filter()
    threshold_filling = rs.threshold_filter(object_min_distance, max_distance)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 1)  # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
    colorizer.set_option(rs.option.min_distance, object_min_distance)
    colorizer.set_option(rs.option.max_distance, max_distance)

    # print(kernel_3)
    # aligned

    align_to = rs.stream.color
    align = rs.align(align_to)
    i = 0
    try:
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            aligned_depth_frame = aligned_frames.get_depth_frame()
            filtered = hole_filling.process(aligned_depth_frame)
            filtered = threshold_filling.process(filtered)
            if not aligned_depth_frame or not color_frame:
                # print('not error')
                continue


            # Convert images to numpy arrays
            depth_image = np.asanyarray(filtered.get_data())


            depth_colormap = np.asanyarray(colorizer.colorize(filtered).get_data())
            depth_image_3d = np.dstack(
                (depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            depth_image_3d_1 = np.asanyarray(colorizer.colorize(filtered).get_data())

            cv2.circle(depth_image_3d_1, (433, 241), 3, (0, 0, 255))
            depth_image_3d_1 = cv2.cvtColor(depth_image_3d_1, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('depth_image_3d',depth_image_3d_1)

            # asdf = aligned_depth_frame.get_distance(x, y)
            # print('distance : '+str(asdf))
            #a=[]
            total=0
            for i in range(x-1,x+2,1):
                for j in range(y-1,y+2,1):
                    #a.append(aligned_depth_frame.get_distance(i, j))
                    total+=aligned_depth_frame.get_distance(i, j)
            mean=total/9
            print(mean)
    finally:

        # Stop streaming

        pipeline.stop()
        print('pipe stop')


# Configure depth and color streams
def stream(width,depth,height,count):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    #0~~65000d = np.where((depth_image_3d > clipping_distance_max) | (depth_image_3d <= clipping_distance_min), grey_color, color_image)

    # Start streaming
    profile= pipeline.start(config)


    print('stream start')
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    # print(depth_scale)

    #value setting (unit:meter)
    shelf_depth=0.45
    object_min_distance = 0.8
    max_distance = object_min_distance + shelf_depth

    object_width=width
    object_height=height
    object_depth=depth
    clipping_distance_in_meters = max_distance # unit : meter
    clipping_distance_in_meters_2= object_min_distance
    clipping_distance_front = object_min_distance / depth_scale
    clipping_distance_behind=max_distance/depth_scale
    blk_size=9
    C=5
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(30,30))
    kernel_3_2=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel_5 = np.ones((5, 5), np.uint8)
    kernel_3 = np.ones((3, 3), np.uint8)
    # print('clipping_distance: =='+str(clipping_distance))
    # print(clipping_distance_2)
    # print(depth_scale)

    #depth sensor setup
    hole_filling = rs.hole_filling_filter()
    threshold_filling=rs.threshold_filter(object_min_distance,max_distance)
    colorizer = rs.colorizer()
    colorizer.set_option(rs.option.visual_preset, 1) # 0=Dynamic, 1=Fixed, 2=Near, 3=Far
    colorizer.set_option(rs.option.min_distance, object_min_distance)
    colorizer.set_option(rs.option.max_distance, max_distance)
    

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
            filtered = threshold_filling.process(filtered)
            if not aligned_depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(filtered.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            # print(depth_image)
            depth_colormap = np.asanyarray(colorizer.colorize(filtered).get_data())




            #Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            #depth_colormap_apply = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
            black=0
            bg_removed = np.where((depth_image_3d > clipping_distance_behind) | (depth_image_3d <= clipping_distance_front),0,color_image)
            # cv2.namedWindow('bg_removed', cv2.WINDOW_AUTOSIZE)
            # cv2.moveWindow('bg_removed', 2000, 100)
            # cv2.imshow('bg_removed',bg_removed)


################image processing



########################################make mask Bitwise_and
            color_image,depth_colormap=mask(color_image,depth_colormap)
#########################################convert to gray
            bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2GRAY)
            color_gray=cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
           
            # add
            HSV = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(HSV)
            #cv2.imshow('v',v)

            depth_colormap_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('aaaaaaaaaaa',depth_colormap_gray)
            # depth_colormap=cv2.cvtColor(depth_colormap,cv2.COLOR_BGR2GRAY)
            #cv2.imshow('1',color_gray)
###########################################blur
            blur = cv2.bilateralFilter(v, 10, 10, 10)
            #blur=cv2.bilateralFilter(color_gray,10,10,10)
            blur_depth=cv2.bilateralFilter(bg_removed,10,10,10)

            blur_depth_colormap=cv2.bilateralFilter(depth_colormap_gray,10,10,10)
            # cv2.imshow('aaaaaaaaaaa',blur_depth_colormap)

            # cv2.namedWindow('blur2', cv2.WINDOW_AUTOSIZE)
            # cv2.moveWindow('blur2', 3000, 100)
            # blur_merge=np.hstack((v, blur, blur_depth))
            # cv2.imshow('blur2', blur_merge)

            #erode=cv2.erode(blur,kernel_3,iterations = 1)#src, filter_d,sigmacolor,sigmaspace
            #cv2.imshow('erode',erode)
            th2= cv2.adaptiveThreshold(blur,240,cv2.ADAPTIVE_THRESH_MEAN_C,\
                                      cv2.THRESH_BINARY,blk_size,5)
            th2_depth = cv2.adaptiveThreshold(blur_depth, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                        cv2.THRESH_BINARY, blk_size, C)
            th2_depth_color = cv2.adaptiveThreshold(blur_depth_colormap, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                              cv2.THRESH_BINARY, blk_size, C)


            # th2_comp=np.hstack((th2,th2_depth))

            # cv2.namedWindow('th3', cv2.WINDOW_AUTOSIZE)
            # cv2.moveWindow('th3', 2000, 400)
            # th3_merge = np.hstack((th2, th2_depth))
            # cv2.imshow('th3', th3_merge)

#########dil
            #erosion = cv2.erode(th2, kernel_3, iterations=1)
            #dilate_color=cv2.dilate(erosion, kernel_5, iterations=1)
            #cv2.imshow('dilation',dilate_color)

            edges=cv2.Canny(th2,200,240)
            edges_depth=cv2.Canny(th2_depth,200,240)

            # cv2.namedWindow('edges_comp4', cv2.WINDOW_AUTOSIZE)
            # cv2.moveWindow('edges_comp4', 4000,800)
            # edges_comp = np.hstack((edges, edges_depth))
            # cv2.imshow('edges_comp4', edges_comp)

            morphol = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_5)
            morphol_depth = cv2.morphologyEx(edges_depth, cv2.MORPH_CLOSE, kernel_5)
            # cv2.namedWindow('mpor', cv2.WINDOW_AUTOSIZE)
            # cv2.moveWindow('mpor', 4000,1000)
            # morphol_merge = np.hstack((morphol, morphol_depth))
            # cv2.imshow('mpor', morphol_merge)


#########################contour

            contours, hierarchy = cv2.findContours(morphol, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            total=0
            for cnt in contours:
                #epsilon = 0.05 * cv2.arcLength(cnt, True)
                #approx = cv2.approxPolyDP(cnt, epsilon, True)
                #print(len(approx))

                #cv2.drawContours(color_image, [approx], 0, (0, 255, 255), 2)
		            #large box width= 220~240 height=110~130
                x, y, w, h = cv2.boundingRect(cnt) #middlie box boundary 145<w<180 85<h<115
                if 140<w<180 and 80<h<110:
                    x_center = int(x + w / 2)
                    y_center = int(y + h / 2)
                    dist_to_center = aligned_depth_frame.get_distance(x_center, y_center)
                    #cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    #cv2.circle(color_image, (x, y), 3, (0, 0, 255))
                    if object_min_distance<dist_to_center <object_min_distance+object_depth+0.1:
                        total+=2
                    elif object_min_distance+object_depth<dist_to_center <max_distance:
                        total+=1

                    # ('x,y:{0},{1}'.format(x,y))

                    # cv2.circle(color_image, (x,y), 3, (0, 0, 255))

                    # print(dist_to_center)

            #print('object 갯수:'+ str(total))


            # contours_d, hierarchy_d = cv2.findContours(th2_depth, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # # print('asdfasdf'+str(hierarchy))
            # # print('asdfasdf'+str(hierarchy_d))
            # for cnt in contours_d:
            #     epsilon = 0.02 * cv2.arcLength(cnt, True)
            #     approx = cv2.approxPolyDP(cnt, epsilon, True)
            #     # print(len(approx))
            #
            #     # cv2.drawContours(depth_colormap, [approx], 0, (0, 0, 255), 1)
            #     x, y, w, h = cv2.boundingRect(cnt) #output x,y,w,h
            #     #if w>100 and h>40:
            #      #   cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), (0, 255, 0), 2)





            #conerharis

            # coners=cv2.goodFeaturesToTrack(edges,80,0.01,10)
            #
            # #print(type(coners))
            # if isinstance(coners, np.ndarray):
            #     coners = np.int32(coners)
            # if isinstance(coners,np.ndarray):
            #
            #     for coner in coners:
            #         x,y=coner[0]
            #         cv2.circle(edges,(x,y),5,(0,0,255),1,cv2.LINE_AA)

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
                cv2.waitKey(1)
            count = total
            i += 1
            cv2.waitKey(1)
            
            if i%20==1:

                cv2.imwrite('img1/' +'_'+ str(i)+'.png', color_image)
                # cv2.imwrite('img1/' +'_depth'+ str(i)+'.png', depth_colormap)

            # elif i==30:
            #     break
            # print(i)
            #
            # elif i==30:





    #  image, light
    finally:

        # Stop streaming

        pipeline.stop()
        print('pipe stop')
        return count
class object():
    count=0
    width=0
    height=0
    depth=0

if __name__== "__main__":
    video_stream = True
    excute=True
    # color_path = NowDate.strftime('%Y-%m-%d_%H%M_' + '.png')
    small_box=object()
    middle_box=object()
    large_box=object()

    small_box_bool = isinstance(small_box, object)
    middle_box_bool = isinstance(middle_box, object)
    large_box_bool = isinstance(large_box, object)
    # print(a)
    if middle_box_bool:

        middle_box.width,middle_box.depth,middle_box.height,middle_box.count = 0.27,0.18,0.15,0 #unit:cm
        stream(middle_box.width,middle_box.depth,middle_box.height,middle_box.count)
        # alignDepth2Color()
        # get_distance(438, 290)
        # stream(ob1.depth,ob1.height,ob1.width,ob1.count)
### image process


    #
    # elif b:
    #     ob2.depth,ob2.height,ob2.width,ob2.count = 19,9,22,0
    #
    #     stream(ob2.depth, ob2.height, ob2.width, ob2.count)
    # elif c:
    #     ob3.depth,ob3.height,ob3.width,ob3.count = 26,21.5,39.5,0
    #     stream(ob3.depth, ob3.height, ob3.width, ob3.count)





#windowName = 'Drawing'
#path = './img/_21.png'

#img = cv2.imread(path, cv2.IMREAD_COLOR)

#cv2.namedWindow(windowName)



