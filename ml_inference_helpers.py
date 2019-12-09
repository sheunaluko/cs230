import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
keras = tf.keras 
import time 
import util as u
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button, RadioButtons


import ml_helpers as ml 

# ML inference helpers 


# load the desired model 
def get_model(model_name) :
    print("Loading model: {}".format(model_name))
    # returns a compiled model
    model = keras.models.load_model(model_name,custom_objects={'IoU' : ml.IoU } ) 
    return model 


def load_model(mod) : 
    mod = "models/{}".format(mod) 
    model = get_model(mod +".h5") 
    info  = np.load(mod + "_history.npy", allow_pickle=True) 
    return model, info.item()


def convert_bb(bb) : 
    bb = 512*bb 
    # need to convert to appropriate shapes 
    pt = (bb[0], bb[1])
    w  = bb[2] - bb[0]
    h  = bb[3] - bb[1]
    return pt,w,h 

# DISPLAY SOME DATA AND ITS BOUNDING BOX 
def show_pred(x,y,index) : 
    nb_imshow(x[index].astype(int),bb=y[index])
    
    
def draw_bb_on_plot(ax,bb,color='lime') : 
    pt,w,h = convert_bb(bb) 
    ax.add_patch(Rectangle(pt,w,h,linewidth=1,edgecolor=color,facecolor='none'))

def nb_visualize_pred(im,y,pred) : 
    plt.imshow(im[:,:,1],cmap='gray')
    ax = plt.gca()
    draw_bb_on_plot(ax,pred,'red') # plot prediction in red 
    draw_bb_on_plot(ax,y,'blue')   # plot truth in blue 
    
    
def nb_visualize_preds(ims,ys,preds) : 
    assert len(ims) == len(ys) == len(preds) 
    
    for i in range(len(ims)) : 
        fig, ax = plt.subplots()
        nb_visualize_pred(ims[i],ys[i],preds[i]) 



   