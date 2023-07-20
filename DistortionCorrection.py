import cv2
import numpy as np
import pandas as pd

# Load the distorted dot array image
img = cv2.imread('C:\\Users\\WTA\\Desktop\\target\\0.42Xdot.bmp')

# Define the 3D coordinates of the dot array
col_pt_num = 17
row_pt_num = 40
objp = np.zeros((col_pt_num * row_pt_num, 3), np.float32)
objp[:, :2] = np.mgrid[0:row_pt_num, 0:col_pt_num].T.reshape(-1, 2)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the corners of the dot array
ret, corners = cv2.findCirclesGrid(gray, (row_pt_num, col_pt_num), None)
distort_pt = np.asarray(corners).reshape(col_pt_num*row_pt_num,2)

# Calibrate the camera using the dot array corners and 3D coordinates
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)

# Undistort the image using the camera calibration parameters
undistorted_img = cv2.undistort(img, mtx, dist)
ret1, undistort_corners = cv2.findCirclesGrid(undistorted_img, (row_pt_num, col_pt_num), None)
center_points = np.asarray(undistort_corners).reshape(col_pt_num*row_pt_num,2)

# dataframe
df_distort = pd.DataFrame(distort_pt)
df_distort_row = pd.DataFrame(distort_pt.reshape(17,80))
df_undistort = pd.DataFrame(center_points)
df_undistort_row = pd.DataFrame(center_points.reshape(17,80))

#print(df_distort_row)

df_row1 = df_distort[[0,1]]
df_row2 = df_undistort[[0,1]]

for i in range(2,79,2):
    df_add1 = df_distort_row[[i,i+1]]
    df_add1.columns=[0,1]
    df_add2 = df_undistort_row[[i,i+1]]
    df_add2.columns=[0,1]
    df_distort680 = pd.concat([df_row1,df_add1])
    df_undistort680 = pd.concat([df_row2,df_add2])
df_distort680.columns = ['x','y']
df_undistort680.columns = ['x','y']
distort680 = np.asarray(df_distort680)
undistort680 = np.asarray(df_undistort680)

# origin
df_undistort.columns = ['x', 'y']

origin_x = df_undistort['x'].mean()
origin_y = df_undistort['y'].mean()


# Calculate the average distortion percentage for all points
h, w = gray.shape[::-1]
#print(len(corners))
#print(len(corners[0]))
num_points = 0
distortion_height = []
distortion_percentage = 0

print(corners)
print(distort680[0])


#for i in range(len(corners)):
#    for j in range(len(corners[i])):
#        # Get the distorted and undistorted coordinates of the point
#        pt_dist = corners[i][j]
#        pt_undist= undistort_corners[i][j]
#
#        num_points += 1
#        print(pt_dist)
#        print(pt_undist)
#
#        # Calculate the distortion percentage for this point
#        distortion_percentage += np.linalg.norm(np.array(pt_dist) - np.array(pt_undist)) / np.linalg.norm(np.array(pt_undist)-np.array([[origin_x,origin_y]]))
#        
#    if num_points%row_pt_num == row_pt_num-1:
#        distortion_percentage /= row_pt_num
#        distortion_percentage *= 100
#        distortion_height.append(distortion_percentage)
#        distortion_percentage = 0

for i in range(len(distort680)):
    # Get the distorted and undistorted coordinates of the point
    pt_dist = distort680[i]
    pt_undist= undistort680[i]
    num_points += 1
    # Calculate the distortion percentage for this point
    distortion_percentage += np.linalg.norm(np.array(pt_dist) - np.array(pt_undist)) / np.linalg.norm(np.array(pt_undist)-np.array([[origin_x,origin_y]]))
        
    if num_points%col_pt_num == col_pt_num-1:
        print(num_points)
        print(i)
        print(pt_dist)
        distortion_percentage /= col_pt_num
        distortion_percentage *= 100
        distortion_height.append(distortion_percentage)
        distortion_percentage = 0

print(distortion_height)
distortion = np.mean(distortion_height)
print(f"Distortion percentage: {distortion:.5f}%")