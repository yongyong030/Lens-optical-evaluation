from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2

def dista(centers,a,b):
    dist_rows = list()
    dot = cv2.imread(root.filename)
    dot_line = dot.copy()

    for i in range(0,b):
        rows = centers[1][i*a:(i+1)*a]
        for j in range(0,a-1):
            x1 = rows[j][0][0]
            y1 = rows[j][0][1]
            x2 = rows[j+1][0][0]
            y2 = rows[j+1][0][1]
            dist = ((x2-x1)**2+(y2-y1)**2)**(1/2)
            cv2.line(dot_line, (int(x1), int(y1)), (int(x2), int(y2)),(0,0,255),1,lineType=cv2.LINE_4)
            dist_rows.append(dist)
        distance = sum(dist_rows)/len(dist_rows)
    cv2.imshow('line fitting',dot_line)
    return distance

root = Tk()
root.title('Pixel Resolution 계산기')
root.geometry('600x300')
root.configure(bg='#fdfbfb')
 
def open():
    global my_image # 함수에서 이미지를 기억하도록 전역변수 선언 (안하면 사진이 안보임)
    root.filename = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(
    ('bmp files', '*.bmp'), ('jpg files', '*.jpg'), ('all files', '*.*')))

    Label(root, text=root.filename,pady=10,font=('Helvetica',10,'bold'),relief=RAISED).pack(expand=4) # 파일경로 view


    
def calculate():

    dot = cv2.imread(root.filename)
    dot_line = dot.copy()

    a = IntVar()
    b = IntVar()
    a = int(ent1.get())
    b = int(ent2.get())

    patternSize = [a, b] #점 개수
    centers = cv2.findCirclesGrid(dot, patternSize)
    distance = dista(centers,a,b)

    Label(root, text="평균거리: "+ str(distance)+" px",pady=5,font=('Helvetica',10,'bold'),relief=RAISED).pack(expand=4)
    Label(root, text="Pixel Resolution: "+ str(500/distance)+" um/px",pady=5,font=('Helvetica',18,'bold'),relief=RAISED).pack(expand=4)

    
 

my_btn = Button(root, text='Load Image',background='#ebedee',pady=5,font=('Helvetica',10,'bold'), command=open).pack(expand=5)

Label(root, text='Columns Number',pady=1,font=('Helvetica',10,'bold')).pack(expand=1) # 파일경로 view
ent1 = Entry(root)
ent1.pack(expand=1)
Label(root, text='Rows Number',pady=1,font=('Helvetica',10,'bold')).pack(expand=1) # 파일경로 view
ent2 = Entry(root)
ent2.pack(expand=1)



my_btn1 = Button(root, text='OK',background='#ebedee',pady=5,font=('Helvetica',10,'bold'), command=calculate).pack(expand=5)
root.mainloop()