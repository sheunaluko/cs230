import numpy as np 
import os 
import cv2 

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
def read_image(fn) : 
    im = cv2.imread(fn,-1) 
    return (im.astype(np.int32)-32768).astype(np.int16) 






    







    
    

