# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 13:16:49 2017

@author: admin
"""

from moviepy.editor import VideoFileClip
#from Ipython.display import HTML
import numpy as np
import cv2, os, pickle, glob
from tracker import tracker, window_mask, pTransform, Calibration
from ImageFunctions import Color_thresholding, Sobel_abs_thresholding, Sobel_dir_thresholding

# Set Directories
curfolder = os.getcwd()
calfolder = os.path.join(curfolder,'camera_cal')
outfolder = os.path.join(curfolder,'output_videos')
if not os.path.exists(outfolder):
    os.mkdir(outfolder)
    
def process_image(img):
    pfile = os.path.join(calfolder,'image_distortion_matrices.p')
    calibMats = Calibration(pfile)
    
    # Undistort Image
    undistort_img = calibMats.undistort(img)
    
    # Thresholding
    color_binary = Color_thresholding(undistort_img,(100,255),(50,255))
    grad_x = Sobel_abs_thresholding(undistort_img,orient='x', kernel = 3, thresh=(40,255))
    dir_sobel = Sobel_dir_thresholding(undistort_img, kernel = 3, thresh=(0.7,1.3))
    combined_threshold = np.zeros_like(color_binary)
    combined_threshold[(grad_x == 1) & (dir_sobel == 1) | (color_binary == 1)] = 255
    
    # Perspective Projection
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
    src_pts = [pt1,pt2,pt3,pt4]
    dst_pts = [[200,250],[1000,250],[1000,720],[200,720]]
    # Applying perspective transform
    perspective_transform = pTransform(src_pts,dst_pts)
    warped_thresh_img = perspective_transform.warped(combined_threshold)
    
    # Attempting sliding window search
    window_width = 50
    window_height = 80
    curve_centers = tracker(Mywindow_width=window_width,Mywindow_height= window_height, 
                            Mymargin = 50, My_ym= 21/720, My_xm= 3.7/800, 
                            Mysmooth_factor=15)
    window_centroids = curve_centers.find_window_centroids(warped_thresh_img)
    # Upon finding centroids, we try to fit polynomial curves
    if len(window_centroids) > 0:
        yvals = np.arange(0, warped_thresh_img.shape[0])
        res_yvals = np.arange(warped_thresh_img.shape[0]-window_height/2, 0, -window_height)
        leftx = window_centroids[:,0]
        rightx = window_centroids[:,1]
        left_fit = np.polyfit(res_yvals, leftx, 2)
        left_poly = np.poly1d(left_fit)
        left_fitx = np.int32(left_poly(yvals))
        right_fit = np.polyfit(res_yvals, rightx, 2)
        right_poly = np.poly1d(right_fit)
        right_fitx = np.int32(right_poly(yvals))
        # assigning lane points
        left_lane = np.array(list(zip(np.concatenate((left_fitx - window_width/2,left_fitx[::-1]+window_width/2), axis=0),
                                  np.concatenate((yvals,yvals[::-1]),axis=0))))
        right_lane = np.array(list(zip(np.concatenate((right_fitx - window_width/2,right_fitx[::-1]+window_width/2), axis=0),
                                       np.concatenate((yvals,yvals[::-1]),axis=0))))
        inner_lane = np.array(list(zip(np.concatenate((left_fitx + window_width/2,right_fitx[::-1]-window_width/2), axis=0),
                                       np.concatenate((yvals,yvals[::-1]),axis=0))))
        # Filling the lane points with colours
        road = np.zeros_like(img)
        cv2.fillPoly(road,np.int32([left_lane]),color=[255,0,0])
        cv2.fillPoly(road,np.int32([right_lane]),color=[0,0,255])
        cv2.fillPoly(road,np.int32([inner_lane]),color=[0,255,0])
        road_bkg = np.zeros_like(img)    
        cv2.fillPoly(road_bkg,np.int32([left_lane]),color=[255,255,255])
        cv2.fillPoly(road_bkg,np.int32([right_lane]),color=[255,255,255])
        # Inverse warp image to fit to the actual image
        road_warped = perspective_transform.invWarped(road)
        road_warped_bkg = perspective_transform.invWarped(road_bkg)
        # This is done only to increase the colour intensity of the final image.
        base = cv2.addWeighted(img, 1.0, road_warped_bkg, -1.0, 0.0) 
        result = cv2.addWeighted(base, 1.0, road_warped, 0.5, 0.0)
        
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
    else:
        result = img
    
    return result

input_video = 'harder_challenge_video.mp4'
output_filename = input_video.split('.mp4')[0] + '_tracked.mp4'
output_video = os.path.join(outfolder,output_filename)

clip = VideoFileClip(input_video)
video_clip = clip.fl_image(process_image)
video_clip.write_videofile(output_video, audio=False)