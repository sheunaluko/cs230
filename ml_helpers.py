import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
keras = tf.keras 
import time 
import util as u
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation

# ML helpers 

#https://www.kaggle.com/vbookshelf/keras-iou-metric-implemented-without-tensor-drama

def convert_to_iou_format(y) : 
    """  
    Will convert from [x_min, y_min, x_max, y_max] to [x, y, width, height] 
    """
    return np.array([ y[0,0] , y[0,1] , y[0,2]-y[0,0] , y[0,3]-y[0,1] ] ,ndmin=2) 


def calculate_iou(y_true, y_pred):
    
    """
    Input:
    Keras provides the input as numpy arrays with shape (batch_size, num_columns).
    
    Arguments:
    y_true -- first box, numpy array with format [x, y, width, height, conf_score]
    y_pred -- second box, numpy array with format [x, y, width, height, conf_score]
    x any y are the coordinates of the top left corner of each box.
    
    Output: IoU of type float32. (This is a ratio. Max is 1. Min is 0.)
    
    """

    results = []
    
    for i in range(0,y_true.shape[0]):
    
        # set the types so we are sure what type we are using
        y_true = convert_to_iou_format(y_true.astype(np.float32))
       
        y_pred = convert_to_iou_format(y_pred.astype(np.float32))   


        # boxTrue
        x_boxTrue_tleft = y_true[0,0]  # numpy index selection
        y_boxTrue_tleft = y_true[0,1]
        boxTrue_width = y_true[0,2]
        boxTrue_height = y_true[0,3]
        area_boxTrue = (boxTrue_width * boxTrue_height)

        # boxPred
        x_boxPred_tleft = y_pred[0,0]
        y_boxPred_tleft = y_pred[0,1]
        boxPred_width = y_pred[0,2]
        boxPred_height = y_pred[0,3]
        area_boxPred = (boxPred_width * boxPred_height)


        # calculate the bottom right coordinates for boxTrue and boxPred

        # boxTrue
        x_boxTrue_br = x_boxTrue_tleft + boxTrue_width
        y_boxTrue_br = y_boxTrue_tleft + boxTrue_height # Version 2 revision

        # boxPred
        x_boxPred_br = x_boxPred_tleft + boxPred_width
        y_boxPred_br = y_boxPred_tleft + boxPred_height # Version 2 revision


        # calculate the top left and bottom right coordinates for the intersection box, boxInt

        # boxInt - top left coords
        x_boxInt_tleft = np.max([x_boxTrue_tleft,x_boxPred_tleft])
        y_boxInt_tleft = np.max([y_boxTrue_tleft,y_boxPred_tleft]) # Version 2 revision

        # boxInt - bottom right coords
        x_boxInt_br = np.min([x_boxTrue_br,x_boxPred_br])
        y_boxInt_br = np.min([y_boxTrue_br,y_boxPred_br]) 

        # Calculate the area of boxInt, i.e. the area of the intersection 
        # between boxTrue and boxPred.
        # The np.max() function forces the intersection area to 0 if the boxes don't overlap.
        
        
        # Version 2 revision
        area_of_intersection = \
        np.max([0,(x_boxInt_br - x_boxInt_tleft)]) * np.max([0,(y_boxInt_br - y_boxInt_tleft)])

        iou = area_of_intersection / ((area_boxTrue + area_boxPred) - area_of_intersection)


        # This must match the type used in py_func
        iou = iou.astype(np.float32)
        
        # append the result to a list at the end of each loop
        results.append(iou)
    
    # return the mean IoU score for the batch
    return np.mean(results)


def IoU(y_true, y_pred):
    
    # Note: the type float32 is very important. It must be the same type as the output from
    # the python function above or you too may spend many late night hours 
    # trying to debug and almost give up.
    
    iou = tf.py_func(calculate_iou, [y_true, y_pred], tf.float32)

    return iou 


# load the desired model 
def load_model(model_name) :
    print("Loading model: {}".format(model_name))
    # returns a compiled model
    from keras.models import load_model
    model = load_model(model_name)

# save model 
def save_model(model,name) : 
    model.save(name)  
    
    
# train curve 
def train_curve(h,name) : 
    train_loss = h.history['loss']
    test_loss  = h.history['val_loss']
    epoch_count = range(1, len(train_loss) + 1)

    plt.plot(epoch_count, train_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0,0.2])
    plt.show();
    plt.savefig("models/" + name + ".png")
    
    
# now build a function for graphing these results over time 
def benchmark_bar(results,title) : 
    xs = [r['name'] for r in results]
    ys = [r['time_info']['t_elapsed'] for r in results] 
    
    y_pos = np.arange(len(ys)) 
    
    plt.bar(y_pos, ys, align='center', alpha=0.5)
    plt.xticks(y_pos, xs)
    plt.ylabel('Time (s)')
    plt.title(title)
    plt.show()
    return plt 
    

# create the model 

def get_baseline_conv_model_1() : 
    ## adapted https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
    model = Sequential()

    #conv
    model.add(Conv2D(32, kernel_size=11, input_shape=(512, 512, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=5))
    model.add(Activation('relu'))

    #pool
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

    #conv 
    model.add(Conv2D(128, kernel_size=5))
    model.add(Activation('relu'))
    model.add(Conv2D(32, kernel_size=5))
    model.add(Activation('relu'))

    #pool 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(4, activation = None))
    
    return model 


# results 

def get_results(x,y) : 
    pred = model.predict(x)
    return (pred, y, y-pred , np.mean( (y-pred)**2) ) 
    
    
    
    
# define model RUN FUNCTIONS 

def run_model(data_fraction=0.1,batch_size=1,num_epochs=10,multi_gpu=False,data=False) : 
    
    
    # load the data 
    if (data) : 
        # data was provided 
        print("\n\nUsing provided data") 
        x_train,y_train,x_val,y_val,x_test,y_test = data  
    else :     
        x_train,y_train,x_val,y_val,x_test,y_test = u.data_load(f=data_fraction)
    
    model_name="v0_t" + str(len(x_train)) + "_e" +  str(num_epochs) + "_b" + str(batch_size)
    print("Runing model:: " + model_name) 
    
    # get the model 
    _model = get_baseline_conv_model_1() 
    
    if (multi_gpu) : 
        print("Creating multi GPU model")
        model = keras.utils.multi_gpu_model(_model,gpus=2) 
    else :  
        model = _model 
            
    # compile model using accuracy to measure model performance
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999), loss='mean_squared_error',metrics=[IoU])
    
    # fit model 
    print("Fitting multi_GPU=[{}] model with bs={},epochs={}".format(str(multi_gpu),str(batch_size),str(num_epochs)) ) 
    t_start = time.time() 
    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs)
    t_end = time.time()
    time_info = {'t_start' : t_start , 't_end' : t_end , 't_elapsed' : (t_end-t_start) } 
    
    print("Saving model") 
    save_model(model,"models/{}.h5".format(model_name) ) 
    
    # return a dictionary 
    return {'name' : model_name, 'train_info' : h , 'time_info' : time_info } 



