import numpy as np 
import os 
import cv2 
import matplotlib.pyplot as plt 
import csv 

# prepare the image directories 
image_dir = "images/Images_png/" 
sub_dirs = os.listdir(image_dir) 
sub_dirs.sort() 

# replace each sub_dir with [ sub_dir [file_list] ] 
files = [] 
for d in sub_dirs : 
    sub_files = os.listdir(os.path.join(image_dir,d))
    sub_files.sort() 
    sub_files_fp = [ os.path.join(image_dir,d,x) for x in sub_files ] 
    files.append( [ d , sub_files_fp ] ) 

# files structure should be ready to go :) 




# given a file can we produce a numpy array 
def read_image(fn,window=True) : 
    im = cv2.imread(fn,-1) 
    im =  (im.astype(np.int32)-32768).astype(np.int16) 
    
    if window : 
        folder = fn.split('/')[-2]        
        win = [float(x) for x in dl_info[folder]['DICOM_windows'].split(",")]
        #win = [-1024,3071] 
        #win = [-1350,150]
        print("For fn: {}, using folder: {}, with window: [{},{}]".format(fn,folder,win[0],win[1]))
        im = windowing(im,win) 
              
    return im 

def show_image(im) : 
    plt.imshow(im,cmap='gray')
    plt.show() 

def disp(fn) : 
    show_image(read_image(fn)) 
    

    
def read_dl_info() : 
    info = {} 
    with open('DL_info.csv') as f:
        a = [{k: v for k, v in row.items()}
             for row in csv.DictReader(f, skipinitialspace=True)]
    for d in a : 
        info[d['File_name'][:-8]] = d 
    return info 

dl_info = read_dl_info() 



def windowing2(im,win): 
    im = im.astype(float)
    return np.min(255, np.max(0, 255*(im-win[0])/(win[1]-win[0])))

def windowing(im, win):
    # (https://github.com/rsummers11/CADLab/blob/master/LesaNet/load_ct_img.py) 
    # scale intensity from win[0]~win[1] to float numbers in 0~255
    im1 = im.astype(float)
    im1 -= win[0]
    im1 /= (win[1] - win[0])
    im1[im1 > 1] = 1
    im1[im1 < 0] = 0
    im1 *= 255
    return im1    


    







    
    

