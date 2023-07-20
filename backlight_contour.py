import cv2
import numpy as np

img = cv2.imread('C:/Users/WTA/Desktop/light_comparison/Macro/3W/1/L3_Origin.bmp')
img2 = img.copy()

# 그레이 스케일로 변환 ---①
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 스레시홀드로 바이너리 이미지로 만들어서 검은배경에 흰색전경으로 반전 ---②
_, imthres = cv2.threshold(imgray, 40, 255, cv2.THRESH_BINARY_INV)

# 가장 바깥쪽 컨투어에 대해 모든 좌표 반환 ---③
contour, hierarchy = cv2.findContours(imthres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(contour)
# 각각의 컨투의 갯수 출력 ---⑤
print('도형의 갯수: %d'% (len(contour)))

# 모든 좌표를 갖는 컨투어 그리기, 초록색  ---⑥
cv2.drawContours(img, contour, -1, (0,255,0), 4)


# 컨투어 모든 좌표를 작은 파랑색 점(원)으로 표시 ---⑧
for i in contour:
    for j in i:
        cv2.circle(img, tuple(j[0]), 1, (255,0,0), -1) 

# 결과 출력 ---⑩
cv2.imshow('CHAIN_APPROX_NONE', img)
cv2.imwrite('C:/Users/WTA/Desktop/test1.png',imthres)

cv2.waitKey(0)
cv2.destroyAllWindows()