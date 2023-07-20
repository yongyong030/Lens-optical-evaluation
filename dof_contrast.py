import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the callback function for mouse events
def draw_window(event, x, y, flags, param):
    global ix, iy, drawing, img, window, draw_type
    draw_type = True

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        img_bg = img.copy()
        cv2.imshow("image", img_bg)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (255, 0, 0), 1)
            cv2.imshow("image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img_fi = img.copy()
        cv2.rectangle(img_fi, (ix, iy), (x, y), (0, 0, 255), 1)
        window = (ix, iy, x, y)
        cv2.imshow("image", img_fi)
        gray_values()
        contrast()
        print(contra.shape)
        print(contra.T)
        contra20()
        plot()
        



def draw_line(event, x, y, flags, param):
    global ix, iy, drawing, img, profile, draw_type
    draw_type = False

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        img_bg = img.copy()
        cv2.imshow("image", img_bg)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            img_copy = img.copy()
            cv2.line(img_copy, (ix, iy), (x, iy), (255, 0, 0), 1)
            cv2.imshow("image", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        img_fi = img.copy()
        cv2.line(img_fi, (ix, iy), (x, iy), (0, 0, 255), 1)
        profile = (ix, iy, x)
        cv2.imshow("image", img_fi)
        gray_values()
        contrast()
        print(contra.shape)
        print(contra)
        
        


def gray_values():
    global roi_x1, roi_x2,roi_y1,roi_y2, gray_img, gray_values_y, y_height

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if draw_type == True:
        roi_x1 = min(window[0],window[2])
        roi_x2 = max(window[0],window[2])
        roi_y1 = min(window[1],window[3])
        roi_y2 = max(window[1],window[3])

        gray_values_y = np.array([gray_img[i, roi_x1:roi_x2] for i in range(roi_y1,roi_y2)])
        y_height = np.array(range(roi_y1,roi_y2))

    elif draw_type == False:
        roi_x1 = min(profile[0],profile[2])
        roi_x2 = max(profile[0],profile[2]) 
        roi_y1 = profile[1]
        roi_y2 = profile[1]+1

        gray_values_y = np.array([gray_img[roi_y1, roi_x1:roi_x2]])
    
    print(gray_values_y)
    print(gray_values_y[0].shape)
    print(roi_x2-roi_x1)
    print(y_height)

def contrast():
    global contra, I_max, I_min, threshold
    
    contra = []
    if roi_x2 < 1300:
        threshold = 90
    elif roi_x1 > 1300:
        threshold = 50

    for i in range(roi_y2-roi_y1):
        grval_lin = gray_values_y[i]

        I_max_sum = 0
        count_max = 0
        I_min_sum = 0
        count_min = 0
        
        for j in range(1, roi_x2-roi_x1-1):
            if (grval_lin[j-1]-grval_lin[j])*(grval_lin[j]-grval_lin[j+1]) <= 0 and grval_lin[j] > threshold:
                I_max_sum += grval_lin[j]
                count_max += 1
                    
            elif (grval_lin[j-1]-grval_lin[j])*(grval_lin[j]-grval_lin[j+1]) <= 0 and grval_lin[j] <= threshold:
                I_min_sum += grval_lin[j]
                count_min += 1
                    
        I_max = I_max_sum / count_max
        I_min = I_min_sum / count_min
        con = (I_max-I_min)/(I_max+I_min)*100
        
        contra.append(con)
        
    contra = np.array(contra)

def contra20():
    global dof_range, base_line

    dof_range = contra.max()*0.8



def plot():
    plt.figure(figsize=(roi_y2-roi_y1,100))
    plt.scatter(y_height, contra,color='black',alpha=0.6)
    plt.plot(y_height,np.full((roi_y2-roi_y1,),dof_range),'r',linestyle="--",alpha=0.4,linewidth=3)
    plt.grid(True)
    plt.yticks(np.arange(0,101,10))
    plt.show()

#def sns_plot():
#    sns.set_style('ticks')
#    sns.scatterplot(x=y_height, y=contra)


# Load the image
img = cv2.imread('C:/Users/WTA/Desktop/target/new_dof2.bmp')

# Create a window to display the image
cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)

# Set the callback function for mouse events
cv2.setMouseCallback("image", draw_window)

# Initialize drawing flag, image copy, and window tuple
drawing = False

window = None

# Display the image
cv2.imshow("image", img)



# Wait for a key press
cv2.waitKey(0)

cv2.destroyAllWindows()
