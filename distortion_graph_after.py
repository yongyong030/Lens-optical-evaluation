import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

col_pt_num = 15
row_pt_num = 36
pt_num = col_pt_num*row_pt_num

target_len = 500 #um
mean_dist = 66.91072 #px
pix_res = 7.47264  #um/px

dot = cv2.imread('C:/Users/WTA/Desktop/target/newdot.bmp')
h,w,c = dot.shape

patternSize = [row_pt_num, col_pt_num] #점 개수
centers = cv2.findCirclesGrid(dot, patternSize)

center_points = np.asarray(centers[1]).reshape(pt_num,2)
#print(center_points[row_pt_num])
#print(center_points.shape)

df = pd.DataFrame(center_points)
df.columns = ['x', 'y']

origin_x = df['x'].mean()
origin_y = df['y'].mean()

if col_pt_num%2 == 0 & row_pt_num%2 == 0:
    calib_pt = np.empty((pt_num,2))
    for i in range(0,pt_num):
        calib_pt[i][0] = origin_x - (row_pt_num/2-0.5-i%row_pt_num)*mean_dist
        calib_pt[i][1] = origin_y - (col_pt_num/2-0.5-i//row_pt_num)*mean_dist
elif col_pt_num%2 == 0 & row_pt_num%2 != 0:
    calib_pt = np.empty((pt_num,2))
    for i in range(0,pt_num):
        calib_pt[i][0] = origin_x - (row_pt_num//2-i%row_pt_num)*mean_dist
        calib_pt[i][1] = origin_y - (col_pt_num/2-0.5-i//row_pt_num)*mean_dist

elif col_pt_num%2 != 0 & row_pt_num%2 == 0:
    calib_pt = np.empty((pt_num,2))
    for i in range(0,pt_num):
        calib_pt[i][0] = origin_x - (row_pt_num/2-0.5-i%row_pt_num)*mean_dist
        calib_pt[i][1] = origin_y - (col_pt_num//2-i//row_pt_num)*mean_dist

elif col_pt_num%2 != 0 & row_pt_num%2 != 0:
    calib_pt = np.empty((pt_num,2))
    for i in range(0,pt_num):
        calib_pt[i][0] = origin_x - (row_pt_num//2-i%(row_pt_num+1)*mean_dist)
        calib_pt[i][1] = origin_y - (col_pt_num//2-i//(row_pt_num+1)*mean_dist)

#print(calib_pt)

df_cali = pd.DataFrame(calib_pt)
df_cali.columns = ['x', 'y']
#print(df_cali)

origin = df.copy()
origin['x'] = origin_x
origin['y'] = origin_y

#print(origin)

df_diff = origin - df
ad = (df_diff['x']**2+df_diff['y']**2)**(1/2)
df_cali_diff = origin - df_cali
prd = (df_cali_diff['x']**2+df_cali_diff['y']**2)**(1/2)

distortion_each = (prd-ad)/prd
dist = distortion_each.to_numpy()
dist_row = dist.reshape(row_pt_num,col_pt_num)
#dist_col = dist.reshape(col_pt_num,17)
y = np.linspace(0,7,8)
dist_row_mean = dist_row.mean(axis=1)*100
print(dist_row_mean)
dist_height = [0,1,2,3,4,5,6,7]
#dist_height[8] = (dist_row_mean[0]+dist_row_mean[16])/2
dist_height[7] = (dist_row_mean[0]+dist_row_mean[14])/2
dist_height[6] = (dist_row_mean[1]+dist_row_mean[13])/2
dist_height[5] = (dist_row_mean[2]+dist_row_mean[12])/2
dist_height[4] = (dist_row_mean[3]+dist_row_mean[11])/2
dist_height[3] = (dist_row_mean[4]+dist_row_mean[10])/2
dist_height[2] = (dist_row_mean[5]+dist_row_mean[9])/2
dist_height[1] = (dist_row_mean[6]+dist_row_mean[8])/2
dist_height[0] = dist_row_mean[7]

print(dist_height)
distortion = abs(dist_row).mean()

plt.scatter(dist_height,y)
plt.show()

print(f'Average Distortion : {distortion:.4%}')

#dot_circle = cv2.circle(dot,(80, 17),16,(0,0,255),1)

#cv2.imshow('distortion',dot)
#cv2.waitKey(0)