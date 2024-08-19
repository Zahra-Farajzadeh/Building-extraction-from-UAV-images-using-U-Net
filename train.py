
import numpy as np
import matplotlib.pyplot as plt
import glob
import rasterio
import cv2
import random

from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_score    # jaccard_score = IOU
from sklearn.metrics import recall_score


import tensorflow as tf
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose, BatchNormalization, Activation
from keras.optimizers import Adam


image_shape = (512,512,4)
label_shape = (512,512,1)

epoch = 50
batch_size = 2

num_class = 1  # building

train_RGBD = glob.glob('/content/drive/MyDrive/Data_complete/train/RGBD/Images/*.tif')
train_MAP = glob.glob('/content/drive/MyDrive/Data_complete/train/MAP/Images/*.tif')

train_RGBD.sort()
train_MAP.sort()

path_TRAIN_RGBD = train_RGBD[0:80]
path_TRAIN_MAP = train_MAP[0:80]

path_VAL_RGBD = train_RGBD[80:100]
path_VAL_MAP = train_MAP[80:100]

print('Number of training images: {} '.format(len(path_TRAIN_RGBD)))
print('Number of validation images: {} '.format(len(path_VAL_RGBD)))

print('Number of training labels: {} '.format(len(path_TRAIN_MAP)))
print('Number of validation labels: {} '.format(len(path_VAL_MAP)))

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self,
                 paths_list,
                 batch_size,
                 epoch,
                 target_size=(512, 512),
                 MAP = False,      # different way to read RGBD and Map
                 RGBD=False):
        
        self.epoch = epoch
        self.batch_size = batch_size
        self.paths_list = paths_list
        self.target_size = target_size
        self.n = len(paths_list)
        self.MAP = MAP
        self.RGBD = RGBD
       
      
    
    def __get_input(self, path, target_size):
      # read RGBD images and GT
          #GT
        if self.MAP:
            image = rasterio.open(path).read()    # shape = [number of chennel, height, width], dtype = float32
            image = np.rollaxis(image, 0, 3)
            # if MAPs are in png format
            # image = cv2.imread(path)    # dtype = uint8
            # convert_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)   # conver RGB To Grayscale image , dtype = float32, shape  = [height, width]  
            # image = convert_image[..., np.newaxis]   # shape = [height, width, 1]
            method = 'nearest'
          #RGBD  
        elif self.RGBD:
            image = rasterio.open(path).read()    # shape = [number of chennel, height, width], dtype = float32
            image = np.rollaxis(image, 0, 3)      # shape = [height, width, number of chennel], dtype = float32
            method = 'bilinear'
            # Resize RGBD image and GT
        image = tf.image.resize(image,(target_size[0], target_size[1]), method=method).numpy()
        
        return image

    
    def __get_data(self, batch_path):
        # Generates data containing batch_size samples
        # don't read all images at ones
        
        X_batch = np.asarray([self.__get_input(image_path, self.target_size) for image_path in batch_path])

        return X_batch
    
    def __getitem__(self, index):
        # get the index of list                               # image = [image1, image2, image3, image4]
        i = (index*self.batch_size) % self.n                  # e.g. index = 1  number of training image is 90 => (1x2) % 90 = 2     % return the remainder of dividing
        batch_path = self.paths_list[i:i+self.batch_size]     # batch_path = path_list [2 : 2+2]
        X = self.__get_data(batch_path)        
        return X
    
    def __len__(self):
      # How many images does the custom generator generate?
                                                               # epoch = 50, number of images = 90, batch_size = 2
        return  self.epoch*self.n // self.batch_size           # e.g.  50 * 90 // 2 = 2250    // is Floor division that returns the largest possible integer
    


rgbd_train_gen = CustomDataGen(path_TRAIN_RGBD, batch_size=batch_size, epoch=epoch, RGBD=True)
GT_train_gen = CustomDataGen(path_TRAIN_MAP, batch_size=batch_size, epoch=epoch, MAP=True)

rgbd_val_gen =  CustomDataGen(path_VAL_RGBD, batch_size=batch_size, epoch=epoch, RGBD=True)
GT_val_gen =  CustomDataGen(path_VAL_MAP, batch_size=batch_size, epoch=epoch, MAP=True)

# generate a random digit to show/plot some samples
batch_number = random.randint(0, len(path_TRAIN_RGBD)-1)


# get the random RGBD and MAP (GT)
x = rgbd_train_gen[batch_number]
print('the RGBD batch shape is {}'.format(x.shape))

y = GT_train_gen[batch_number]
print('the Ground Truth batch shape is {}'.format(y.shape))


# # plot using matplotlib
# fig, axes = plt.subplots(1, 3, figsize=(12, 6))
# axes = axes.ravel()

# axes[0].imshow(x[0,..., 0:3])
# axes[0].title.set_text('RGB')
# axes[0].axis('off')

# axes[1].imshow(x[0,..., 3])
# axes[1].title.set_text('nDSM')
# axes[1].axis('off')

# axes[2].imshow(y[0, ... , 0])
# axes[2].title.set_text('Ground Truth')
# axes[2].axis('off')

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)                       # * Conv2D(filters, strides, padding)
    x = BatchNormalization()(x)                                             #Not in the original network. 
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)                                              #Not in the original network
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p  


#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)      # Conv2DTranspose(num_filters, kernel_size, strides, padding)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


#Build Unet using the blocks
def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)                       
    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)


    outputs = Conv2D(n_classes, 1, padding="same", activation='sigmoid')(d4)              #Change the activation based on n_classes


    model = Model(inputs, outputs, name="U-Net")
    return model

model = build_unet(image_shape, n_classes=1)
model.compile(optimizer=Adam(learning_rate = 1e-4), loss='binary_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
     

# model.summary()
# from keras.utils import plot_model
# plot_model(model, show_shapes=True)


history = model.fit(
    x = zip(rgbd_train_gen, GT_train_gen),
    batch_size = batch_size,
    epochs = 5,
    verbose = 1,
    shuffle = True,
    validation_data = zip(rgbd_val_gen, GT_val_gen),
    steps_per_epoch = rgbd_train_gen.n // batch_size,        
    validation_steps = rgbd_val_gen.n // batch_size,
    )


model.save('model_simple_U-Net.h5')

 # "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
