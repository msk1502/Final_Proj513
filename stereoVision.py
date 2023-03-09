import numpy as np
import cv2
import simplestereo as ss


#import images
img_l =
img_r =

#setup stereorig
res_l = #resolution of left camera (width height) 
fx_l = #focal length x of left camera
fy_l = #focal length y of left camer#
x0_l = #principial point offest x of left camera
y0_l = #principial point offest y of left camera
s_l = #axis skew of left camera
intrinsic_l = np.matrix('fx_l, s_l, x0_l; 0, fy_l, y0_l; 0, 0, 1')

res_r = #resolution of right camera (width height) 
fx_r = #focal length x of right camera
fy_r = #focal length y of right camer#
x0_r = #principial point offest x of right camera
y0_r = #principial point offest y of right camera
s_r = #axis skew of right camera
intrinsic_r = np.matrix('fx_r, s_r, x0_r; 0, fy_r, y0_r; 0, 0, 1')

stereo = ss.StereoRig(res_l, res_r, intrinsic_l, intrinsic_r)

#or use simplestereo.calibration.chessboardStereo

#create rectified rig
Rcommon = np.array(',,;,,;,,')
Rect_homography1 = np.array(',,;,,;,,')
Rect_homography2 = np.array(',,;,,;,,`')
rigRect = ss.RectifiedStereoRig(Rcommon, Rect_homography1, Rect_homography2, stereo)

#rectify images
img_l_Rect = rigRect.rectifyImages(img_l,img_r)

#call OpenCV passive stereo algorithms
stereo = cv2.StereoSGBM_create()
disparityMap = stereo.compute(img_l_Rect, img_l_Rect)

#get 3D points
points3D = rigRect.get3DPoints(disparityMap)
ss.points.exportPLY(points3D, 'export.ply', img_l_Rect)

#normalize and color
disparityImg = cv2.normalize()
di