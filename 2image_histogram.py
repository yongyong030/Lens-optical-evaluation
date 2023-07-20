import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2
import os

# 축 없애기
def axis_delete(ax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

def img_mask(grayimage):
    _, imthres = cv2.threshold(grayimage, 40, 255, cv2.THRESH_BINARY_INV)
    return imthres

def plot_graph(image1,image1gray,mask1,image2,image2gray,mask2):

    # 그래프 Layout
    gridsize = (4, 2)
    fig = plt.figure(figsize=(9, 12))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid(gridsize, (2, 0), rowspan=2)
    ax3 = plt.subplot2grid(gridsize, (2, 1), rowspan=2)
    x = np.linspace(0,254,255)

    # Origin
    histr1 = cv2.calcHist([image1gray],[0],mask1,[255],[0,255])
    ax1.plot(histr1,color = 'darkgreen',label = 'Before',linewidth = 3)
    ax1.set_xlim([40,256])
    ax1.set_ylim([0,20000])
    y1 = histr1.reshape(-1)
    ax1.fill_between(x,y1,alpha = 0.3, color = 'darkgreen')

    # After
    histr2 = cv2.calcHist([image2gray],[0],mask2,[255],[0,255])
    ax1.plot(histr2,color = 'lime',label = 'After',linewidth = 3)
    ax1.set_xlim([40,256])
    ax1.set_ylim([0,20000])
    y2 = histr2.reshape(-1)
    ax1.fill_between(x,y2,alpha = 0.3, color = 'lime')

    # Setting
    plt.rcParams['font.family'] = 'Times New Roman'
    ax1.set_title('Light Comparison',fontsize=18)
    ax1.set_ylabel('Counts', fontsize = 14)
    ax1.legend(['Before','After'], fancybox = False,edgecolor = 'black')

    # Images
    ax2.imshow(image1)
    ax2.set_title('Before',y=-0.2,fontsize=14)
    axis_delete(ax2)
    ax3.imshow(image2)
    ax3.set_title('After',y=-0.2, fontsize=14)
    axis_delete(ax3)

def img_readtogray(dir):
    image = img.imread(dir)
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    return image, image_gray

dir_histo = ['L1','L2','L3','L4']

def comparison(optical_system):
    path = f'C:/Users/WTA/Documents/iwta/조명밝기향상인서트 촬상/DF_comparison/{optical_system}'

    for j in range(1,8):
        print(f'{j}번째 샘플')
        
        dir_3W = os.listdir(path+f'/3W/{j}')
        dir_10W = os.listdir(path+f'/10W/{j}')

        _, img_mask1 = img_readtogray(path+f'/3W/{j}/L3_Origin.bmp')
        _, img_mask2 = img_readtogray(path+f'/10W/{j}/L3_After_m.bmp')

        for dir1,dir2,save_name in zip(dir_3W,dir_10W,dir_histo):
            print(dir1,dir2)

            img1, img1gray = img_readtogray(path+f'/3W/{j}/{dir1}')
            mask1 = img_mask(img_mask1)

            img2, img2gray = img_readtogray(path+f'/10W/{j}/{dir2}')
            mask2 = img_mask(img_mask2)
            
            plot_graph(img1,img1gray,mask1,img2,img2gray,mask2)

            plt.savefig(path+f'/histogram/{j}/{save_name}.png')

#comparison('Macro')
comparison('Micro')