import numpy as np 
import os 
import cv2 
import matplotlib.pyplot as plt 
import csv 
import json
from collections import Counter 
from matplotlib.patches import Rectangle
import math


# manage windows/linux stuff
import platform 
_os = platform.system() 
if _os == "Linux"  : 
    fdelim = "/"
elif _os == "Darwin" : 
    fdelim = "/"
elif _os == "Windows" :     
    fdelim  = "\\" 
else : 
    print("unrecognized os!") 

# prepare the image directories 
image_dir = "images" + fdelim + "Images_png" + fdelim 
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
def read_image(fn,with_win=False,bb=True,verbose=True) : 
    im = cv2.imread(fn,-1) 
    im =  (im.astype(np.int32)-32768).astype(np.int16) 
    
    # only look up the window if with_win is False 
    win = with_win or [float(x) for x in dl_info[fn]['DICOM_windows'].split(",")]
    if verbose : 
        print("For fn: {}, using window: [{},{}]".format(fn,win[0],win[1]))
        
    im = windowing(im,win) 
        
    # now get the bounding box as well 
    if bb :
        _bb = [round(float(x)) for x in dl_info[fn]['Bounding_boxes'].split(',') ] 
        return (im , _bb, win )
    else : 
        return im 


def gen_neighbor_names(fn) : 
    tok = fn.split(fdelim) # ['images', 'Images_png', '000001_01_01', '103.png']
    slice_tok = tok[-1].split(".")
    left_num  = "{:03d}".format(int(slice_tok[0]) - 1)
    right_num = "{:03d}".format(int(slice_tok[0]) + 1)
    left_fn   = fdelim.join(tok[0:3]) + fdelim + left_num + ".png" 
    right_fn   = fdelim.join(tok[0:3]) + fdelim + right_num + ".png" 
    return (left_fn, right_fn) 


def read_image_and_neighbors(fn,verbose=True) : 
    # should be able to assume that the slices are available on either side 
    lfn, rfn = gen_neighbor_names(fn) 
    
    # first we read the main image and get the window and bounding box 
    mim, bb, win = read_image(fn,verbose=verbose) 
    
    # now we will read the left and right images using the same window and w/o bb
    lim = read_image(lfn,with_win=win,bb=False,verbose=verbose)
    rim = read_image(rfn,with_win=win,bb=False,verbose=verbose)

    # are going to produce a matrix (512,512,3) 
    slices = np.zeros( (512,512,3 ) ) 

    slices[:,:,0] = lim
    slices[:,:,1] = mim 
    slices[:,:,2] = rim 
    
    return (slices, np.array(bb,ndmin=2)) 


def show_image(im,bb=False) : 
    plt.imshow(im,cmap='gray')
    plt.ion()
    plt.show() 
    
    # if bounding box will also draw the bb 
    if bb.any() : 
        # need to convert to appropriate shapes 
        pt = (bb[0], bb[1])
        w  = bb[2] - bb[0]
        h  = bb[3] - bb[1]
        print("Using bb coords: ({},{}),{},{}".format(pt[0],pt[1],w,h))
        plt.gca().add_patch(Rectangle(pt,w,h,linewidth=1,edgecolor='lime',facecolor='none'))

    plt.draw()
    plt.pause(0.001) # non blocking 
        
def disp(fn,bb=False) : 
    im, bb, win = read_image(fn)  # read the image and the bounding box 
    if bb :
        show_image(im,bb=bb) 
    else : 
        show_image(im) 

def disp_loop() :
    plt.figure()
    for folder in files  :
        for f in folder[1] :
            im = read_image(f)
            plt.imshow(im,cmap='gray')
            plt.pause(0.1)
            plt.draw()

def test_show() :
    disp("images/Images_png/000001_03_01/088.png", bb=True)
    
def show_liver(num) :
    disp(liver_lesions[num]['File_name'],bb=True)
    
def read_json_labels() :             
    with open('text_mined_labels_171_and_split.json') as json_file: 
        data = json.load(json_file)
        return data 
    
json_labels = read_json_labels() 

def get_index_of_term(t) : 
    return json_labels['term_list'].index(t)
        

def search_for_term(term, to_search) :   # term is actually an index here  
    matches = [] 
    for i,val in enumerate(to_search) : 
        # each val here is a list [x, x2, x3.. ] 
        # if 'term' is in this list then we add it to matches 
        if term in val : 
            matches.append([i,val])
    return matches
    
    
def read_dl_info_vector() : 
    
    #function for modifying map object after generated 
    def transform_map(m) : 
        #fixes fname 
        tok = m['File_name'].split("_")  
        m['File_name'] = image_dir +  "_".join(tok[0:3]) + fdelim + tok[-1]            
        return m 
        
    with open('DL_info.csv') as f:
        a = [{k: v for k, v in row.items()}
             for row in csv.DictReader(f, skipinitialspace=True)]
        return [transform_map(x) for x in a] 
    
dl_info_vector = read_dl_info_vector() 

def read_dl_info() : 
    info = {} 
    a  = read_dl_info_vector() 
    for d in a : 
        info[d['File_name']] = d 
    return info 

dl_info = read_dl_info() 

def select_lesion_idxs(s) : 
    return [ dl_info_vector[x] for x in s ] 


# -- liver dev (8 is liver) 
liver_slices = search_for_term(8, json_labels['train_relevant_labels'])
liver_lesion_tmp_idx = [ x[0] for x in liver_slices ] 
liver_lesion_idx = [ json_labels['train_lesion_idxs'][i] for i in liver_lesion_tmp_idx ] 

liver_lesions = select_lesion_idxs(liver_lesion_idx) 
coarse_types = Counter([x['Coarse_lesion_type'] for x in liver_lesions]) 

# -- 

def generate_term_specific_set(train_val_test, term) : 
    labs       = search_for_term(term, json_labels['{}_relevant_labels'.format(train_val_test)])
    labs_idx   = [ x[0] for x in labs ]
    lesion_idx = [ json_labels['{}_lesion_idxs'.format(train_val_test)][i] for i in labs_idx  ] 
    lesions    = select_lesion_idxs(lesion_idx) 
    coarse_types = Counter([x['Coarse_lesion_type'] for x in lesions]) 
    return { "lesions" : lesions , 
             "coarse_types" : coarse_types , 
             "lesion_idx"  : lesion_idx , 
             "labs_idx" : labs_idx , 
             "labs" : labs } 


def build_partitioned_dataset(lesions,name,num_parts) : 

    num_per_part = math.floor(len(lesions)/num_parts) 
    print("{} per part".format(num_per_part))
    
    for k in range(0,num_parts) : 
        part = lesions[num_per_part*k:num_per_part*(k+1)]
        part_number = str(k+ 1)
        print("Building part #{} with {} lesions".format(part_number,len(part)))
        build_dataset(part,name+"_part_"  + part_number)
        
    if len(lesions) % num_per_part != 0 : 
        part = lesions[num_per_part*num_parts:len(lesions)]
        part_number = str(k+ 1)
        print("Building part #{} with {} lesions".format(part_number,len(part)))
        build_dataset(part,name+"_part_"  + part_number)
    
    #done 
    

def build_dataset(lesions,name) : 
    num_lesions  = len(lesions) 
    xs = np.zeros( (num_lesions, 512,512,3 ) ) 
    ys = np.zeros( (num_lesions, 1, 4 ) ) 
    
    print("Generating data set...") 
    
    for i,v in enumerate(lesions) : 
        
        #if ( i % 200 == 0 ) : 
        #  print("On index: " + str(i)) 
        
        # get the filename of the lesion 
        fn = lesions[i]['File_name']
        
        # TODO get the data -- WILL wrap INSIDE TRY CATCH and if error then will 
        # print the FILENAME so I can explore where the error happened
        slices,bounding_box = read_image_and_neighbors(fn,verbose=False) 
        
        # append the data
        xs[i,:,:,:] = slices 
        ys[i,:,:]   = bounding_box 
        
    # at this point data set should be built 
    # will write the data to numpy binary file 
    #print("Saving xs...")
    np.save(name + '_xs',xs)
    #print("Saving ys...")
    np.save(name + '_ys',ys)
    print("Done!") 

    

# for extracting bounding box: the values given are just the indece
# manually review liver lesions 
# DRAW the bounding box on the liver lesions and verify with KEY_SLICES examples
# bottom left pixel x,y and top right pixel x,y 


# first pass: grab image on either side and skip interpolation 
# later -> figure out interpolation 


# write documentaiton for how to load data 


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


    

print("Loaded util") 

def reload() : 
    import importlib 
    import sys
    importlib.reload(sys.modules['util'])



    
    


