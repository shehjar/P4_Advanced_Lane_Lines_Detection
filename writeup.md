**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Undistortion_Example.jpg "Undistorted"
[image2]: ./output_images/undistorted_0.jpg "Road Undistorted"
[image3]: ./output_images/thresholding_output_0.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines_0.jpg "Warp Example"
[image5]: ./output_images/warped_4.jpg "Fit Visual"
[image6]: ./output_images/final_result_0.jpg "Output"
[video1]: ./output_videos/project_video_tracked.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md)

The writeup.md is written in as comprehensible manner as possible. The template provided above was used to edit the final one here.

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I have a special python file created for the calibration calculation - [undistort.py](./undistort.py)
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. The corners within an image is detected by an OpenCV function called `cv2.findChessboardCorners(gray, (nx, ny), None)` where `gray` is the image of the chessboard in grayscale and `(nx,ny)` represent the shape of the number of expected corners in the image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![undistorted image][image1]

To use the distortion matrices for the processing the images, I had created a class called `Calibration()` within [tracker.py](./tracker.py).
###Pipeline (single images)
The pipeline for single images are demonstrated below. The python file used for processing single images is [LaneFinder.py](./LaneFinder.py)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![undistorted road][image2]

The image comparison here is very fine, but one can see the slight differences in the way the image is cropped at the edges.

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
The colour and gradient thresholds were defined as functions in the [ImageFunctions.py](./ImageFunctions.py) file. I used a combination of color and gradient thresholds, specifically the S and the L channel for the colour thresholds and the gradient along x and slope thresholds to generate a binary image (thresholding steps at lines 49 through 53 in [LaneFinder.py](./LaneFinder.py)).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![Binary image][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perspective transform is defined as a class called `pTransform()` in the [tracker.py](./tracker.py) file. The class is defined by the source and destination points of an image and these are found separately. Once the instance of the class is created, it has a function called `warped()` which projects the image as an input to a bird's eye view. the class also contains `invWarped()` function which brings the image back into the original camera frame of reference view.  I chose the hardcode the source and destination points in the following manner:
```
ht,wd = image.shape
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
dst_pts = [[200,250],[1000,250],[1000,720],[200,720]]
```
The source points (`src_pts`) were found though ratios of the image size given above. The parameters were changed according to a particular image and tested if the trapezoid fit best after the perspective transform.
The destination points (`dst_pts`) were directly added without any ratios, but the point positions were imagined to be the best configuration of the birds-eye view of the image.

The source and the destination points were given into the class instance below and the image is warped using the class object `perspective_transform`
```
perspective_transform = pTransform(src_pts,dst_pts)
warped_img = perspective_transform.warped(image)

```
This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 570, 468      | 200, 250      |
| 715, 468      | 1000, 250     |
| 1065, 698     | 1000, 720     |
| 259, 698      | 200, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For the identification of lane lines, I implemented the sliding window technique using convolutions. In the program, this is a part of a class called `tracker()` and within the function of `find_window_centroids`. The line is detected if a convolution of the histogram of the part of the image gives a positive signal. The signal is further analysed to distinguish between the left and the right lanes and the center of the signal is taken as output. Once, the centroids were found, the curve fitting was done as a function `x = f(y)` where `y` represents the pixels in the height direction of the image and `x` represents the width of the image. The fitting of my lane lines was done with a 2nd order polynomial like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is calculated for the single images and the video files in the two different python files ([LaneFinder.py](./LaneFinder.py), [video_lane_detector.py](./video_lane_detector.py). The below code calculates the curvature -
```
# Calculate the radius of curvature
res_yvals_ym = res_yvals*curve_centers.ym_per_pix
leftx_xm = leftx*curve_centers.xm_per_pix
rightx_xm = rightx*curve_centers.xm_per_pix
curve_fit_left = np.polyfit(res_yvals_ym, leftx_xm,2)
curve_fit_right = np.polyfit(res_yvals_ym, rightx_xm,2)
ht_m = ht*curve_centers.ym_per_pix
curve_radius_left = np.power(1+(2 * curve_fit_left[0] * ht_m + curve_fit_left[1])**2,3/2)/np.abs(2*curve_fit_left[0])
curve_radius_right = np.power(1+(2 * curve_fit_right[0] * ht_m + curve_fit_right[1])**2,3/2)/np.abs(2*curve_fit_left[0])
```
the y values are taken as per the window height dimension and the x values are the centroid values for left and the right lanes. The pixel data is transformed into meters data by multiplying the ratio of `ym_per_pix` and `xm_per_pix`. The new data goes through the 2nd order polynomial curve fitting and the resulting coefficients gives us the curvature radius - `curve_radius_left` and `curve_radius_right`. There is no averaging done for the two and they are individually reported in the final output of the image.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The resulting image from adding colours in the detected lane areas needs to be inverse warped so that it can be put on the original undistorted image. The inverse warping is also done via the class `pTransform` through the function `invWarped()`. Eventually, the lane markings and the road are coloured in the final image, as shown below

![Final Output][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video_tracked.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The perspective transform implementation took a bit of time, as I had initially tried to make it robust by first calculating the vanishing point. However, I was unable to get the other source points in this case and therefore later resorted to manual entry of such points which may differ from image to image. The next challenge was getting the lane markings identified properly. This also depended on the kind of thresholding done previously and the bird's eye view of the road, which is again parameter dependent, which was done manually. The final image was being skewed due to the presence of a nearby car. I tried to crop the part of the car by changing the destination points in the perspective transform, to make it work. Also the calculation of the radius of curvature is done through mapped parameters like pixels per meter, which can also be inaccurate.

The pipeline that I created did well for the project video, but did not fare well with the challenge videos as the perspective transformation of the image needs more robustness.
