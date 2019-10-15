from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import Sequential


'''I am using a combination of 2 tutorials:
1- https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/ (How to use VGG16 on Keras)
2- https://keras.io/applications/ (Examples on how to cut the model off at a specific layer)
'''

base_model = VGG16(weights='imagenet')
#Extracting features from end of 2nd block of layers in VGG16, size is  (None, 14, 14, 512)
pre_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_conv3').output)

#Testing it on a picture
img_path = 'cat.jpg' #Test image
img = image.load_img(img_path, target_size=(224, 224)) #The model expects 224x224x3 inputs by default
x = image.img_to_array(img) #Turning image into an array
x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2])) # (1 image , 224 , 224, 3)
x = preprocess_input(x) # Preprocessing input for vgg16

vgg_16_2_features = pre_model.predict(x) #shape of feature tensor is (1, 14, 14, 512)

''' Now that we have imported the first two layers of vgg16, let's build the rest of the model. '''

model = Sequential()

#TODO rest of the model