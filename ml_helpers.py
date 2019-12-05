import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
keras = tf.keras 
import time 
import util as u
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Activation, Dropout


#vgg16
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions 



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

    

# save model 
def save_model(model,name) : 
    model.save(name)  
    
    
# train curve 
def train_curve(h,name=False) : 
    train_loss = h['loss']
    test_loss  = h['val_loss']
    epoch_count = range(1, len(train_loss) + 1)

    plt.plot(epoch_count, train_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Test Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0,0.2])
    
    # conditionally save 
    if name : 
        plt.savefig("models/" + name + ".png")
        
    plt.show();

def jupyter_show_png(fname) : 
    from IPython.display import Image 
    return Image(filename=fname)
    
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

def get_baseline_vgg_model_1(dropout=False) : 
    
    if (dropout) : 
        print("\nUsing dropout: {}\n".format(dropout)) 
    
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (512,512,3) )
    # should (later) try both layers block4_conv3 and block3_conv3
    vgg_extraction = vgg_model.get_layer('block4_conv3').output # (1, 64, 64, 512)
    
    X = Conv2D(32, kernel_size=5)(vgg_extraction)
    X = Activation('relu')(X)
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    

    #pool
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #conv 
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    X = Conv2D(32, kernel_size=5)(X)
    X = Activation('relu')(X)

    #pool 
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #flatten and dense 
    X = Flatten()(X)
    boxes = Dense(4, activation = None)(X)
    
    model = keras.Model(vgg_model.input, boxes) 
    
    #make sure to free all of the vgg layers 
    for layer in vgg_model.layers : 
        layer.trainable = False 

    return model     


def get_baseline_vgg_model_3(dropout=False) : 
    
    if (dropout) : 
        print("\nUsing dropout: {}\n".format(dropout)) 
    
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (512,512,3) )

    vgg_extraction = vgg_model.get_layer('block3_conv3').output # ?
    
    X = Conv2D(32, kernel_size=5)(vgg_extraction)
    X = Activation('relu')(X)
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    

    #pool
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #conv 
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    X = Conv2D(32, kernel_size=5)(X)
    X = Activation('relu')(X)

    #pool 
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #flatten and dense 
    X = Flatten()(X)
    boxes = Dense(4, activation = None)(X)
    
    model = keras.Model(vgg_model.input, boxes) 
    
    #make sure to freeze all of the vgg layers 
    for layer in vgg_model.layers : 
        layer.trainable = False 

    return model   


def baseline_for_vgg_co_model(dropout=False) : 

    model_in = keras.Input(shape=(512, 512, 515))  # takes in added vgg features 3 image channels + 512 vgg channels
    
    if (dropout) : 
        print("\nUsing dropout: {}\n".format(dropout)) 
        
    #conv
    X = Conv2D(32, kernel_size=11)(model_in) 
    X = Activation('relu')(X)
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)

    #pool
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)
        
    #conv 
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    X = Conv2D(32, kernel_size=5)(X)
    X = Activation('relu')(X)

    #pool 
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    X = Flatten()(X)
    boxes = Dense(4, activation = None)(X) 
    
    return keras.Model(model_in,boxes)

def vgg_sub_model() : 
    
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (512,512,3))
    layers_to_drop = ["block4_pool" , "block5_pool"]                   
                      
    model = Sequential()
    for layer in vgg_model.layers : 
        if (layer.name not in layers_to_drop ) :  
            #layer.trainable = True 
            model.add(layer)
            
#    model.add(keras.layers.UpSampling2D(size=(8, 8), data_format="channels_last", interpolation='nearest'))
                      
    return model 

def vgg_sub_model_2() : 
    
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (512,512,3))
    layers_to_drop = ["block4_pool" , "block5_pool"]       
    layers_to_freeze  = ["block1_conv1" , "block1_conv2" , "block2_conv1" , "block2_conv2" ] 
                      
    model_in = keras.Input(shape=(512,512,3))
    
    X = model_in 
    for layer in vgg_model.layers : 
        if (layer.name not in layers_to_drop ) :  
            if (layer.name in layers_to_freeze) : 
                print("Freezing layer: " + layer.name) 
                layer.trainable = False
            X = layer(X) 
        else : 
            print("Dropping layer: " + layer.name)
            
    return (model_in, X ) 

def get_baseline_vgg_model_no_pool(dropout=False) : 
    
    if (dropout) : 
        print("\nUsing dropout: {}\n".format(dropout)) 
    
    (model_in, vgg) = vgg_sub_model_2()  
    
    X = Conv2D(32, kernel_size=11)(vgg)
    X = Activation('relu')(X)
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    
    #pool
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #conv 
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    X = Conv2D(32, kernel_size=5)(X)
    X = Activation('relu')(X)

    #pool 
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #flatten and dense 
    X = Flatten()(X)
    boxes = Dense(4, activation = None)(X)
    
    model = keras.Model(model_in, boxes) 
    
    return model   



                      
def vgg_co_baseline_model(dropout=False) : 
    model_in = tf.keras.Input(shape=(512,512,3)) 
    
    # get the sequential vgg_sub_model 
    vgg_sub = vgg_sub_model() 
    # get baseline model 
    baseline = baseline_for_vgg_co_model(dropout=dropout) 
    
    # extract vgg features 
    vgg_features = vgg_sub(model_in) #(should be 64,64,512) 
    
    # upsample these features to match the input shape (times 8) 
    vgg_upsampled = keras.layers.UpSampling2D(size=(8, 8), data_format="channels_last", interpolation='nearest')(vgg_features)
    
    # now concatenate the upsampled vgg feature channels to the original input 
    new_input = keras.layers.Concatenate(axis=-1)([model_in,vgg_upsampled]) 
    
    # and then run the new_input through the baseline model 
    boxes = baseline(new_input) 
    
    # and then return the final model 
    return keras.Model(model_in, boxes) 
                      

def get_baseline_vgg_model_2(dropout=False) : 
    
    if (dropout) : 
        print("\nUsing dropout: {}\n".format(dropout)) 
    
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (512,512,3) )

    vgg_extraction = vgg_model.get_layer('block2_conv2').output # ?
    
    X = Conv2D(32, kernel_size=5)(vgg_extraction)
    X = Activation('relu')(X)
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    

    #pool
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #conv 
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    X = Conv2D(32, kernel_size=5)(X)
    X = Activation('relu')(X)

    #pool 
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #flatten and dense 
    X = Flatten()(X)
    boxes = Dense(4, activation = None)(X)
    
    model = keras.Model(vgg_model.input, boxes) 
    
    #make sure to freeze all of the vgg layers 
    for layer in vgg_model.layers : 
        layer.trainable = False 

    return model   



def get_baseline_vgg_model_4tr(dropout=False) : 
    
    if (dropout) : 
        print("\nUsing dropout: {}\n".format(dropout)) 
    
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (512,512,3) )

    vgg_extraction = vgg_model.get_layer('block4_conv3').output # ?
    
    X = Conv2D(32, kernel_size=5)(vgg_extraction)
    X = Activation('relu')(X)
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    

    #pool
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #conv 
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    X = Conv2D(32, kernel_size=5)(X)
    X = Activation('relu')(X)

    #pool 
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #flatten and dense 
    X = Flatten()(X)
    boxes = Dense(4, activation = None)(X)
    
    model = keras.Model(vgg_model.input, boxes) 
    

    return model   

def get_vgg_model(layer,dropout=False,trainable=False) :    
    if (dropout) : 
        print("\nUsing dropout: {}\n".format(dropout)) 
    
    vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = (512,512,3) )

    vgg_extraction = vgg_model.get_layer(layer).output # ?
    
    X = Conv2D(32, kernel_size=5)(vgg_extraction)
    X = Activation('relu')(X)
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    
    #pool
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #conv 
    X = Conv2D(128, kernel_size=5)(X)
    X = Activation('relu')(X)
    X = Conv2D(32, kernel_size=5)(X)
    X = Activation('relu')(X)

    #pool 
    X = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(X)
    if (dropout) : 
        X = Dropout(dropout)(X)

    #flatten and dense 
    X = Flatten()(X)
    boxes = Dense(4, activation = None)(X)
    
    model = keras.Model(vgg_model.input, boxes) 
    
    #make sure to freeze all of the vgg layers 
    if trainable : 
        print("Vgg backbone will be trainable")
    else : 
        print("Vgg backbone will NOT be trainable")
        for layer in vgg_model.layers : 
            layer.trainable = False 

    return model   

# results 

def get_results(x,y) : 
    pred = model.predict(x)
    return (pred, y, y-pred , np.mean( (y-pred)**2) ) 
    
    
    
    
# define model RUN FUNCTIONS 

def run_model(data_fraction=0.1,
              batch_size=1,
              num_epochs=10,
              multi_gpu=False,
              data=False,
              learning_rate=0.001,
              learning_rate_decay = False, 
              dropout = False , 
              tensorboard=False,
              save=False,
              describe=False, 
              model_id=None) : 

    callbacks = None 
    if tensorboard : 
        print("\nUsing tensorboard\n")
        import datetime 
        log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks = [tensorboard_callback] 

    
    # load the data 
    if (data) : 
        # data was provided 
        print("\nUsing provided data\n") 
        x_train,y_train,x_val,y_val,x_test,y_test = data  
    else :     
        x_train,y_train,x_val,y_val,x_test,y_test = u.data_load(f=data_fraction)
    
    if (model_id == 'baseline') : 
        model_name= "v0_t" + str(len(x_train)) + "_e" +  str(num_epochs) + "_b" + str(batch_size) + "_lr" + str(learning_rate) 
        # get the model 
        _model = get_baseline_conv_model_1() 
        
    elif (model_id == 'baseline_vgg' ) : 
        model_name= "vBVGG_t" + str(len(x_train)) + "_e" +  str(num_epochs) + "_b" + str(batch_size) + "_lr" + str(learning_rate) 
        # get the model 
        _model = get_baseline_vgg_model_1(dropout=dropout) 
        
    elif (model_id == 'baseline_vgg_block3' ) : 
        model_name= "vBVGG3_t" + str(len(x_train)) + "_e" +  str(num_epochs) + "_b" + str(batch_size) + "_lr" + str(learning_rate) 
        # get the model 
        _model = get_baseline_vgg_model_3(dropout=dropout) 
        
    elif (model_id == 'baseline_vgg_block2' ) : 
        model_name= "vBVGG2_t" + str(len(x_train)) + "_e" +  str(num_epochs) + "_b" + str(batch_size) + "_lr" + str(learning_rate) 
        # get the model 
        _model = get_baseline_vgg_model_2(dropout=dropout) 
    
    elif (model_id == 'vgg_co_baseline' ) : 
        model_name= "vVCB" + str(len(x_train)) + "_e" +  str(num_epochs) + "_b" + str(batch_size) + "_lr" + str(learning_rate) 
        # get the model 
        _model = vgg_co_baseline_model(dropout=dropout) 
                
        
    elif (model_id == 'baseline_vgg_block4tr' ) :
        model_name= "vBVGG4tr_t" + str(len(x_train)) + "_e" +  str(num_epochs) + "_b" + str(batch_size) + "_lr" + str(learning_rate) 
        _model = get_baseline_vgg_model_4tr(dropout=dropout)  
   
    elif (model_id == 'baseline_vgg_no_pool' ) :
        model_name= "vBVNP_t" + str(len(x_train)) + "_e" +  str(num_epochs) + "_b" + str(batch_size) + "_lr" + str(learning_rate) 
        _model = get_baseline_vgg_model_no_pool(dropout=dropout)  
        

        
    else :
        raise Exception("Must specify model id!") 
            
    if (learning_rate_decay) : 
        model_name += "_lrd" 
    if (dropout) : 
        model_name = "{}_d{}".format(model_name,dropout) 
    
    if (multi_gpu) : 
        print("Creating multi GPU model")
        model = keras.utils.multi_gpu_model(_model,gpus=2) 
    else :  
        model = _model 
            
    print("\nRuning model:: {}\n".format(model_name))
   
            
    # compile model 
    if learning_rate_decay :
        print("\nUsing rate decay\n") 
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                         decay=learning_rate/num_epochs,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss='mean_squared_error',metrics=[IoU])
    else : 
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                         beta_1=0.9,
                                                         beta_2=0.999),
                      loss='mean_squared_error',metrics=[IoU]) 
        
        
    # describe the model 
    if (describe) : 
        describe_model(model) 
    
    # fit model 
    print("\nFitting multi_GPU=[{}] model with bs={},epochs={},lr={}\n".format(str(multi_gpu),str(batch_size),str(num_epochs),learning_rate))
    t_start = time.time() 
    
    h = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=num_epochs,callbacks=callbacks) 
        
    t_end = time.time()
    time_info = {'t_start' : t_start , 't_end' : t_end , 't_elapsed' : (t_end-t_start) } 
    
    if save : 
        print("\nSaving model") 
        save_model(model,"models/{}.h5".format(model_name) ) 
    
    # return a dictionary with the model instance as well 
    return {'name' : model_name, 'train_info' : h , 'time_info' : time_info , "model" : model } 


def describe_model(model) : 
    for i, layer in enumerate(model.layers):
       print("{}, {} , {}".format(i, layer.name,layer.trainable))
    model.summary()


