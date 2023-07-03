# %%
import os
import numpy as np 
import pandas as pd 
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import winsound

# %%
for dirname, _, filenames in os.walk('../dataset/'):    
    files = {}
    for filename in filenames:        
        files[filename] = os.path.join(dirname, filename)
print(files)

# Open classes.txt
with open(files['classes.txt']) as file: 

   classes = file.read()

def str_to_list(line):
    line = line.replace('\n','')
    line = line.replace('classes = ','')
    line = eval(line)
    return line
    
classes = str_to_list(classes)
print(classes)

def notification():
    for _ in range(5):
        winsound.Beep(500, 200)
        winsound.Beep(1200, 200)
        winsound.Beep(500, 200)
        winsound.Beep(1200, 200)

# %%
labels_path = '../dataset/labels.npy'
signals_path = '../dataset/signals.npy'
snrs_path = '../dataset/snrs.npy'

labels = np.load(labels_path, mmap_mode = 'r')
signals = np.load(signals_path, mmap_mode = 'r')
snrs = np.load(snrs_path, mmap_mode = 'r')

print(signals.shape)
print(labels.shape)
print(snrs.shape)


# %%
# Dataset is to big, need to lowering the traiuing size

x_train, x_test, y_train, y_test = train_test_split(signals, labels, train_size=0.125, stratify=labels, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, stratify=y_train, random_state = 42)
print(x_train.shape)
print(x_val.shape)
notification()

# %%
from keras import layers
from tensorflow.keras.utils import to_categorical
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.layers import Input, Dropout, Add, Dense, Reshape, Activation
from keras.layers import BatchNormalization, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.optimizers import Adam

# %%
class Residual_block:
    kernel_size = 3
    strides = 1
    padding = 'same'
    data_format = "channels_last"

    def __init__(self, x, x_shortcut, filters):
        self.x = x
        self.filters = filters
        self.x_shortcut = x_shortcut

    def unit(self):
        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(self.x)
        x = Activation('relu')(x)
        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(x)
        x = Activation('linear')(x)
        # add skip connection
        if x.shape[1:] == self.x_shortcut.shape[1:]:
            x = Add()([x, self.x_shortcut])
        else:
            raise Exception('Skip Connection Failure!')
        return x

class Convolution_block:
    kernel_size = 1
    strides = 1
    padding = 'same'
    data_format = "channels_last"

    def __init__(self, x, filters):
        self.x = x
        self.filters = filters

    def unit(self):
        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(self.x)
        x = Activation('linear')(x)
        return x
    
def residual_stack(x, filters):
    x = Convolution_block(x, filters)
    print('x')
#     print(x.shape)
    print(x)
    x = x.unit()
    print('xunit')
#     print(x.shape)
    print(x)
    
    x_shortcut = x
    x = Residual_block(x, x_shortcut, filters)
    x = x.unit()
    x_shortcut = x
    x = Residual_block(x, x_shortcut, filters)  
    x = x.unit()
    
    # max pooling layer
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
#     print('Residual stack created')
    return x

def ResNet(input_shape, classes):   
    # create input tensor
    x_input = Input(input_shape)
    x = x_input
    # residual stack
    num_filters = 40
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    
    # output layer
    x = Flatten()(x)
    x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
    x = Dropout(.5)(x)
    x = Dense(128, activation="selu", kernel_initializer="he_normal")(x)
    x = Dropout(.5)(x)
    x = Dense(classes , activation='softmax', kernel_initializer = glorot_uniform(seed=0))(x)
    
    # Create model
    model = Model(inputs = x_input, outputs = x)
#     print('Model ResNet created')
    return model

save_model = False
save_history = False

# create directory for model weights
if save_model is True:
    weights_path = input("Name model weights directory: ")
    weights_path = "data/weights/" + weights_path

    try:
        os.mkdir(weights_path)
    except OSError:
        print ("Creation of the directory %s failed" % weights_path)
    else:
        print ("Successfully created the directory %s " % weights_path)
    print('\n')
    

# create directory for model history
if save_history is True:
    history_path = input("Name model history directory: ")
    history_path = "data/model_history/" + history_path

    try:
        os.mkdir(history_path)
    except OSError:
        print ("Creation of the directory %s failed" % history_path)
    else:
        print ("Successfully created the directory %s " % history_path)
    print('\n')

adm = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# set number of epochs
num_epochs = int(input('Enter number of epochs: '))

# set batch size
batch = 32

# configure weights save

if save_model is True:
    filepath= weights_path + "/{epoch}.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode="auto")
    callbacks_list = [checkpoint]
else:
    callbacks_list = []

# %%
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")

# %%
for i in [x_train, y_train, x_val, y_val]:
    i = tf.convert_to_tensor(i, np.float32)


# %%
model = ResNet((1024, 2), 24)
model.compile(optimizer=adm, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train, epochs = num_epochs, batch_size = 16, callbacks=callbacks_list, validation_data=(x_val, y_val))
notification()

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
tf.config.list_physical_devices('GPU') 


