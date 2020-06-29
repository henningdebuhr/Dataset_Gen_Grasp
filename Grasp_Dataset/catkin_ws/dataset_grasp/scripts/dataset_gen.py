#!/usr/bin/env python

import os.path
import rospy
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
import time
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
#rom matplotlib.patches import Rectangle
import numpy as np
import darknet_ros_msgs
import scipy as sp
import math
import canny_edge_det as ced



import numpy as np
def rotate(point, origin, degrees):
    radians = np.deg2rad(degrees)
    x,y = point
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy

def rotate_image(image, angle):
  image_center = (150,150)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


bridge = CvBridge()
fig = plt.subplots(2,2)
plt.ion()
####################################################################
#   Adjust this for path and bagfilename
####################################################################
dataset_directory = '/home/hdb/Grasp_Dataset/dataset/'
bag_file_directory = '/home/hdb/Grasp_Dataset/bagfiles/'
#bagfilename = "Alu40mm_bc"
bagfilename = "Alu20_cc"
# ros bag file read 
# put in folder with .py or adjust path
bag_to_open = bag_file_directory + bagfilename+".bag"

#print bag_to_open
####################################################################

bag = rosbag.Bag(bag_to_open)
# timerange for find correnponding detection # tenmil+tenmil= 20 msec
tenmil = rospy.Duration(0, 1000000000).to_sec()
#imagecounter
for topic, msg, t in bag.read_messages(topics=['/darknet_ros/detection_image']):
    color_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    

for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw']):
    depth_img = bridge.imgmsg_to_cv2(msg, '16UC1')
    depth_img = np.uint8(depth_img)

count = 0
# for all the images in bagfile
for topic, msg, t in bag.read_messages(topics=['/camera/aligned_depth_to_color/image_raw']):
    print ("Checked image %i" %count)
    cv_img = bridge.imgmsg_to_cv2(msg, '16UC1')
    imgtime = msg.header.stamp.to_sec()
    print (imgtime)
    count += 1    
    # for all images find object detection pair
    for topic, msg, t in bag.read_messages(topics=['/darknet_ros/bounding_boxes']):
        # timing for detection and img time
        dettime = msg.header.stamp.to_sec() 
        print ("correnponding detection@time:")
        print (dettime)

        if  imgtime-tenmil <= dettime and imgtime+tenmil>= dettime:

            # for each detected obj in image
            for box in msg.bounding_boxes:
                # show boxes in img , bounding box and rect box for cut
                #cv2.rectangle(cv_img, (box.xmin, box.ymin), (box.xmax, box.ymax), color=(255, 255, 255), thickness=3)
                #cv2.rectangle(cv_img, (((box.xmin+box.xmax)/2)-100, ((box.ymin+box.ymax)/2)-100), (((box.xmin+box.xmax)/2)+99, ((box.ymin+box.ymax)/2)+99), color=(0, 0, 0), thickness=3)
                # crop image, at the border we get black areas also after rotation
                crop_img = cv_img[((box.ymin+box.ymax)/2)-150: ((box.ymin+box.ymax)/2)+149, ((box.xmin+box.xmax)/2)-150:((box.xmin+box.xmax)/2)+149]

                # when rotating the image x times one degree, it gets blurry so always rotate original image once for x degrees
                for x in range(119):
                    rot = x*3
                    a = np.mean(crop_img, axis=0)
                    b = np.mean(crop_img, axis=1)
                    # buffer to get rid of black spaces after rotation 
                    blank = np.full((300,300),(np.mean(a,axis=0)+np.mean(b,axis=0)/2))
                    for i in xrange(0, crop_img.shape[0]):
                        for j in xrange(0, crop_img.shape[1]):
                            blank[i,j]= crop_img[i,j]

                    
                    # adjust for each img/rosbag
                    poi1=(50,100)
                    poi2=(150,100)
                    points=[poi1, poi2]
                    # rotations for img and anotations
                    # rotate the cut again to get rid of black spaces
                    rot_img = rotate_image(blank, rot)
                    rot_img = rot_img[50:250, 50:250]
                    poi1_to_img = rotate(poi1, (100,100), rot )
                    poi2_to_img = rotate(poi2, (100,100), rot )
                    
                    # Save rotated image for dataset
                    mat_file_name = dataset_directory + bagfilename + "_" + str(x)+".npy"
                    np.save(mat_file_name, rot_img)
                    # Save anotaions - points for training
                    text_file_name = dataset_directory + bagfilename + "_" + str(x) +".txt"
                    text_file = open(text_file_name, "wb")
                    annotations = np.uint8((poi1_to_img,poi2_to_img))
                    #print annotations
                    np.savetxt(text_file, annotations,fmt='%d')
                    #text_file = open("Output.txt", "r")
                    #lines = np.loadtxt('Output.txt', dtype=int)
                    #print lines
                    text_file.close()

                    rot_img = np.load(mat_file_name)
                    # print rot_mat # check for orig val in mm
                    # images to plot must be unint8
                    plot_img = np.uint8(rot_img)
                    # canny edge algo this should look like what the ai sees after conv filters
                    edges = cv2.Canny(plot_img,5,255)
                    # show poi & annotation in img
                    cv2.circle(plot_img,(int(poi1_to_img[0]),int(poi1_to_img[1])), 5, 200, 3)
                    cv2.circle(plot_img,(int(poi2_to_img[0]),int(poi2_to_img[1])), 5, 200, 3)
                    #plots
                    plt.subplot(221),plt.imshow(color_img)
                    plt.title('Original Color Image'), plt.xticks([]), plt.yticks([])
                    plt.subplot(222),plt.imshow(depth_img,cmap = 'bone')
                    plt.title('Original Depth Image'), plt.xticks([]), plt.yticks([])
                    plt.subplot(223),plt.imshow(plot_img,cmap = 'bone')
                    plt.title('200x200 Crop for Dataset with POI'), plt.xticks([]), plt.yticks([])
                    plt.subplot(224),plt.imshow(edges,cmap = 'bone')
                    plt.title('Edges/Convolutional'), plt.xticks([]), plt.yticks([])
                    plt.draw()
                    #raw_input("Press Enter to continue...")#debug
                    plt.pause(0.001) # adjust speed
                    #plt.pause(20) # use this to read the poi coords from img with mouse from the plot
                    plt.clf()  

bag.close()