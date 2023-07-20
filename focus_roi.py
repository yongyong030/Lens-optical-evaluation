import cv2

# 이미지를 불러옵니다.
img = cv2.imread("C:/Users/WTA/Documents/iwta/focusvarisample/000.bmp")

# 이미지의 가로, 세로 크기를 구합니다.
height, width = img.shape[:2]

# 이미지를 그레이스케일로 변환합니다.
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 이미지를 블러처리하여 노이즈를 제거합니다.
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

# 이미지에서 에지를 검출합니다.
edge_img = cv2.Canny(blur_img, 100, 200)

# 이미지에서 컨투어를 찾습니다.
contours, hierarchy = cv2.findContours(edge_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어를 정렬합니다.
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# 가장 큰 컨투어를 선택합니다.
largest_contour = contours[0]

# 컨투어를 감싸는 사각형을 구합니다.
x, y, w, h = cv2.boundingRect(largest_contour)

# ROI를 추출합니다.
roi = img[y:y+h, x:x+w]

# ROI를 출력합니다.
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
