import sys
import json
#from pathlib import Path
import numpy as np
import pandas as pd
import re
#import matplotlib.pyplot as plt
import xlrd
import cv2
import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
#from keras import backend as K


df = pd.read_excel('C:/Users/akhil/OneDrive/Desktop/Perron_phenotype-GSU-training.xlsx', sheet_name='Total Database');
X_train = [];
Y_train = [];
for root, dirs, files in os.walk('D:/Masters/Introduction to Machine Learning/Data/Training'):
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(root, file));
            if img is not None:
                img_name = re.findall(r'\d+', os.path.basename(os.path.join(root, file)));
                y = df.loc[df['strain'] == int(img_name[0])];
                y = np.asarray(y);
                y = np.take(y, range(1, 30));
                Y_train.append(y);
                X_train.append(img);

X_train = np.asarray(X_train);
Y_train = np.asarray(Y_train);

X_test = [];
Y_test = [];
for root, dirs, files in os.walk('D:/Masters/Introduction to Machine Learning/Data/Validation'):
    for file in files:
        if file.endswith('.jpg'):
            img = cv2.imread(os.path.join(root, file));
            if img is not None:
                img_name = re.findall(r'\d+', os.path.basename(os.path.join(root, file)));
                y = df.loc[df['strain'] == int(img_name[0])]
                y = np.asarray(y);
                y = np.take(y, range(1, 30));
                Y_test.append(y);
                X_test.append(img);

X_test = np.asarray(X_test);
Y_test = np.asarray(Y_test);

'''
Y_train = pd.read_excel('C:/Users/akhil/OneDrive/Desktop/Training.xlsx', sheet_name='Sheet1');
Y_train = np.asarray(Y_train);
#for i in range(0, 27):
    #Y_train[i] = (Y_train[i] - min(Y_train[i])) / (max(Y_train[i]) - min(Y_train[i]));

Y_test = pd.read_excel('C:/Users/akhil/OneDrive/Desktop/Testing.xlsx', sheet_name='Sheet1');
Y_test = np.asarray(Y_test);
#for i in range(0, 8):
    #Y_test[i] = (Y_test[i] - min(Y_test[i])) / (max(Y_test[i]) - min(Y_test[i]));
'''

dtype_mult = 255.0;
X_train = X_train.astype('float32') / dtype_mult;
X_test = X_test.astype('float32') / dtype_mult;


model = Sequential();
# Conv1 256 256 (3) => 254 254 (32)
model.add(Conv2D(32, (3, 3), input_shape= (256, 256, 3))); # in layer 1 you need to specify input shape this is not needed in subsequent layers
model.add(Activation('relu'));

# Conv2 254 254 (32) => 252 252 (32)
model.add(Conv2D(32, (3, 3)));
model.add(Activation('relu'));

# Pool1 252 252 (32) => 126 126 (32)
model.add(MaxPooling2D(pool_size=(2, 2))); # the CONV CONV POOL structure is popularized in during ImageNet 2014
model.add(Dropout(0.25)); # this thing called dropout is used to prevent overfitting

# Conv3 126 126 (32) => 124 124 (64)
model.add(Conv2D(64, (3, 3)));
model.add(Activation('relu'));

# Conv4 124 124 (64) => 122 122 (64)
model.add(Conv2D(64, (3, 3)));
model.add(Activation('relu'));

# Pool2 122 122 (64) => 61 61 (64)
model.add(MaxPooling2D(pool_size=(2, 2)));
model.add(Dropout(0.25));

'''
# Conv5 125 125 (64) => 123 123 (128)
model.add(Conv2D(128, (3, 3)));
model.add(Activation('relu'));

# Conv6 123 123 (128) => 120 120 (128)
model.add(Conv2D(128, (4, 4)));
model.add(Activation('relu'));

# Pool3 120 120 (128) => 40 40 (128)
model.add(MaxPooling2D(pool_size=(3, 3)));
model.add(Dropout(0.25));
'''

# FC layers 61 61 (64) =>  
model.add(Flatten()); # to turn input into a 1 dimensional array

# Dense1  => 512
model.add(Dense(512));
model.add(Activation('linear'));
model.add(Dropout(0.5));


# Dense2 512 => 256
model.add(Dense(256));
model.add(Activation('linear'));
model.add(Dropout(0.5));


# Dense3 256 => 29
model.add(Dense(29));
model.add(Activation('linear'));
model.add(Dropout(0.25));

'''
# Dense4 256 => 29
model.add(Dense(29));
model.add(Activation('linear'));
model.add(Dropout(0.25));
'''

optimizer = keras.optimizers.Adam(); # Adam is one of many gradient descent formulas and one of the most popular

#loss = tf.losses.huber_loss(Y_test, model.predict(X_test), weights=1.0, delta=1.0, scope=None, loss_collection=tf.GraphKeys.LOSSES, reduction=Reduction.SUM_BY_NONZERO_WEIGHTS)

model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics = ['mean_absolute_error']);

#nb_epoch = 4; # number of iterations to train on
#batch_size = 1; # process entire image set by chunks of 1

model.fit(X_train, Y_train, batch_size = 1, nb_epoch = 4, validation_data = (X_test, Y_test), shuffle = 'True');

print(model.predict(X_test));

# to save model architecture
outfile = open('D:/Masters/Introduction to Machine Learning/Project/model.json', 'w');
json.dump(model.to_json(), outfile);
outfile.close();

# to save model weights
model.save_weights('D:/Masters/Introduction to Machine Learning/Project/weights.h5');

'''
def eval(loc1, loc2): # loc1 => Excel_sheet_location        loc2 => images_location
    
    dataFrame = pd.read_excel(loc1, sheet_name='Total Database');
    X = [];
    Y = [];
    for root, dirs, files in os.walk(loc2):
        for file in files:
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(root, file));
                if img is not None:
                    img_name = re.findall(r'\d+', os.path.basename(os.path.join(root, file)));
                    y = df.loc[df['strain'] == int(img_name[0])]
                    y = np.asarray(y);
                    y = np.take(y, range(1, 30));
                    Y.append(y);
                    img = cv2.resize(image, (256, 256));
                    X.append(img);

    X = np.asarray(X);
    Y = np.asarray(Y);
    
    # to load model architecture
    infile = open('D:/Masters/Introduction to Machine Learning/Project/model.json');
    model = keras.models.model_from_json(json.load(infile));
    infile.close();
    
    # to load model weights
    model.load_weights('D:/Masters/Introduction to Machine Learning/Project/weights.h5');
    
    model.evaluate(x = X, y = Y, batch_size = 5, verbose=1, sample_weight = None);
    
    return
'''

'''
get_17th_layer_output = K.function([model.layers[0].input], [model.layers[17].output]);
layer_output = get_17th_layer_output([X_test])[0];
print(layer_output);
'''
