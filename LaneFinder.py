# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 10:25:55 2017

@author: admin
"""

import cv2, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle
from ImageFunctions import Color_thresholding, Sobel_abs_thresholding, Sobel_dir_thresholding
from tracker import tracker, window_mask, pTransform, Calibration

curfolder = os.getcwd()
calfolder = os.path.join(curfolder,'camera_cal')
srcfile = os.path.join(curfolder,'test_images')
outfolder = os.path.join(curfolder, 'output_images')

#Get distortion matrices
pfile = os.path.join(calfolder,'image_distortion_matrices.p')
calibMats = Calibration(pfile)
dist = calibMats.dist
mtx = calibMats.mtx

# Get all the images
stImagePaths = glob.glob(os.path.join(srcfile,'straight_lines*.jpg'))
testImagePaths = glob.glob(os.path.join(srcfile,'test*.jpg'))
imagePaths = stImagePaths + testImagePaths

# Browsing through all the images - 
for idx,path in enumerate(imagePaths):
    img = cv2.imread(path)
    # Undistort Image
    undistort_img = calibMats.undistort(img)
    # Thresholding
    color_binary = Color_thresholding(undistort_img,(100,255),(35,255))
    grad_x = Sobel_abs_thresholding(undistort_img,orient='x', kernel = 3, thresh=(40,255))
    dir_sobel = Sobel_dir_thresholding(undistort_img, kernel = 3, thresh=(0.7,1.3))
    combined_threshold = np.zeros_like(color_binary)
    combined_threshold[(grad_x == 1) & (dir_sobel == 1) | (color_binary == 1)] = 255
    # Creating image out of the thresholding functions
    thresh_img = cv2.merge((np.uint8(combined_threshold),np.uint8(combined_threshold),np.uint8(combined_threshold)))
    # Saving thresholded output
    outfile = os.path.join(outfolder,'thresholding_output_'+str(idx)+'.jpg')
    cv2.imwrite(outfile,thresh_img)

    # Perspective Projection
    img_copy = undistort_img.copy()
    ht,wd = combined_threshold.shape
    top_offset = 3
    bottom_offset = 22
    base_ratio = 0.63
    height_ratio = 0.32
    top_level_ratio = 0.65
    r = 0.18
    trap_top = int(top_level_ratio*ht)
    trap_height = int(height_ratio*ht)
    trap_bottom = trap_top + trap_height
    trap_base = int(base_ratio*wd)
    trap_top_length = int(trap_base*r)
    pt1 = [int(wd/2-trap_top_length/2)+top_offset, trap_top]
    pt2 = [int(wd/2+trap_top_length/2)+top_offset, trap_top]
    pt3 = [int(wd/2+trap_base/2)+bottom_offset, trap_bottom]
    pt4 = [int(wd/2-trap_base/2)+bottom_offset, trap_bottom]

    # Select Source and Destination Points
    src_pts = [pt1,pt2,pt3,pt4]
    #dest_pts = [[int(wd/2-trap_base/2)+bottom_offset, 0], [int(wd/2+trap_base/2)+bottom_offset, 0],
    #            [int(wd/2+trap_base/2)+bottom_offset, trap_bottom],[int(wd/2-trap_base/2)+bottom_offset, trap_bottom]]
    dst_pts = [[200,250],[1000,250],[1000,720],[200,720]]
    perspective_transform = pTransform(src_pts,dst_pts)
    #M = cv2.getPerspectiveTransform(np.float32(src_pts),np.float32(dst_pts))
    #Minv = cv2.getPerspectiveTransform(np.float32(dst_pts),np.float32(src_pts))
    warped_img = perspective_transform.warped(img_copy)

    # Visualization of Perspective Transform
    pts_orig = np.array(src_pts)
    pts_orig = pts_orig.reshape((-1,1,2))
    cv2.polylines(img_copy,[pts_orig],True,(255,0,0),thickness=10)
    pts_warped = np.array(dst_pts)
    pts_warped = pts_warped.reshape((-1,1,2))
    cv2.polylines(warped_img,[pts_warped],True,(255,0,0),thickness=10)
    # plotting and saving
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    img_copy_RGB = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)
    ax1.imshow(img_copy_RGB)
    ax1.set_title('Undistorted Image with source points drawn', fontsize=20)
    warped_img_RGB = cv2.cvtColor(warped_img,cv2.COLOR_BGR2RGB)
    ax2.imshow(warped_img_RGB)
    ax2.set_title('Warped result with dest. points drawn', fontsize=20)
    persp_outfile = os.path.join(outfolder,'warped_straight_lines_'+str(idx)+'.jpg')
    plt.savefig(persp_outfile)

    # Attempting sliding window search
    warped_thresh_img = perspective_transform.warped(combined_threshold)
    window_width = 50
    window_height = 80
    curve_centers = tracker(Mywindow_width=window_width,Mywindow_height= window_height, 
                            Mymargin = 50, My_ym= 21/720, My_xm= 3.7/800, 
                            Mysmooth_factor=15)
    window_centroids = curve_centers.find_window_centroids(warped_thresh_img)

    # Plotting the final graph
    # If we found any window centers
    if len(window_centroids) > 0:

        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped_thresh_img)
        r_points = np.zeros_like(warped_thresh_img)
    
        # Go through each level and draw the windows 	
        for level in range(0,len(window_centroids)):
            # Window_mask is a function to draw window areas            
            l_mask = window_mask(window_width,window_height,warped_thresh_img,window_centroids[level][0],level)
            r_mask = window_mask(window_width,window_height,warped_thresh_img,window_centroids[level][1],level)
            # Add graphic points from window mask here to total pixels found 
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template) # create a zero color channle 
        template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
        warpage = np.array(cv2.merge((warped_thresh_img,warped_thresh_img,warped_thresh_img)),np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
 
    # If no window centers found, just display orginal road image
    else:
        #output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
        output = warped_thresh_img
    # fit the lane boundaries to the left , right center positions found
    yvals = np.arange(0, warped_thresh_img.shape[0])
    res_yvals = np.arange(warped_thresh_img.shape[0]-window_height/2, 0, -window_height)
    leftx = window_centroids[:,0]
    rightx = window_centroids[:,1]
    left_fit = np.polyfit(res_yvals, leftx, 2)
    left_poly = np.poly1d(left_fit)
    left_fitx = np.int32(left_poly(yvals))
    #left_fitx = 
    right_fit = np.polyfit(res_yvals, rightx, 2)
    right_poly = np.poly1d(right_fit)
    right_fitx = np.int32(right_poly(yvals))
    
    left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2,left_fitx[::-1]+window_width/2), axis=0),
                                  np.concatenate((yvals,yvals[::-1]),axis=0))))
    right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2,right_fitx[::-1]+window_width/2), axis=0),
                                   np.concatenate((yvals,yvals[::-1]),axis=0))))
    inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width/2,right_fitx[::-1]-window_width/2), axis=0),
                                   np.concatenate((yvals,yvals[::-1]),axis=0))))
    road = np.zeros_like(img)
    cv2.fillPoly(road,np.int32([left_lane]),color=[255,0,0])
    cv2.fillPoly(road,np.int32([right_lane]),color=[0,0,255])
    
    # Save file
    warpage = np.array(cv2.merge((warped_thresh_img,warped_thresh_img,warped_thresh_img)),np.uint8)
    output = cv2.addWeighted(warpage, 1, road, 0.5, 0.0)
    output_text1 = 'Left Lane = {0[0]:.3e}y^2 + {0[1]:.3e}y + {0[2]:.3e}'.format(left_fit)#+' from left and '+ str(round(window_centroids[0,1]))+ ' from right'
    output_text2 = 'Right Lane = {0[0]:.3e}y^2 + {0[1]:.3e}y + {0[2]:.3e}'.format(right_fit)
    cv2.putText(output,output_text1,(int(window_centroids[0,0])+25,ht-50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(output,output_text2,(int(window_centroids[0,0])+25,ht-100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    warped_outfile = os.path.join(outfolder,'warped_'+str(idx)+'.jpg')
    cv2.imwrite(warped_outfile, output)
    
    cv2.fillPoly(road,np.int32([inner_lane]),color=[0,255,0])
    road_bkg = np.zeros_like(img)    
    cv2.fillPoly(road_bkg,np.int32([left_lane]),color=[255,255,255])
    cv2.fillPoly(road_bkg,np.int32([right_lane]),color=[255,255,255])
    
    road_warped = perspective_transform.invWarped(road)
    road_warped_bkg = perspective_transform.invWarped(road_bkg)
    
    base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0) 
    result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)
    
    # Calculate the radius of curvature
    res_yvals_ym = res_yvals*curve_centers.ym_per_pix
    leftx_xm = leftx*curve_centers.xm_per_pix
    rightx_xm = rightx*curve_centers.xm_per_pix
    curve_fit_left = np.polyfit(res_yvals_ym, leftx_xm,2)
    curve_fit_right = np.polyfit(res_yvals_ym, rightx_xm,2)
    ht_m = ht*curve_centers.ym_per_pix
    curve_radius_left = np.power(1+(2 * curve_fit_left[0] * ht_m + curve_fit_left[1])**2,3/2)/np.abs(2*curve_fit_left[0])
    curve_radius_right = np.power(1+(2 * curve_fit_right[0] * ht_m + curve_fit_right[1])**2,3/2)/np.abs(2*curve_fit_left[0])
    
    # Calculate the offset of the car on the road
    camera_center = (left_fitx[-1] + right_fit[-1])/2
    center_diff = (camera_center-warped_thresh_img.shape[1]/2)*curve_centers.xm_per_pix
    if center_diff <= 0:
        side_pos = 'right'
    else:
        side_pos = 'left'
        
    # Put the text on the resulting image
    cv2.putText(result, 'Radius of Curvature = '+str(round(curve_radius_left,3))+'(m) Left and '+str(round(curve_radius_right,3))+'(m) Right',
                (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(result, 'Vehicle is '+str(abs(round(center_diff,3)))+'m '+ side_pos +' of center', 
                (50,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    final_file = os.path.join(outfolder,'final_result_'+str(idx)+'.jpg')
    cv2.imwrite(final_file,result)