from __future__ import print_function  
import keras  
from keras.datasets import mnist  
from keras.models import Sequential  
from keras.layers import Dense, Dropout, Flatten  
from keras.layers import Conv2D, MaxPooling2D  
import time
import pandas as pd
import numpy as np
import os
from tensorflow.python.lib.io import file_io

def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())
            

job_dir='gs://wz_bucket/temp/'
batch_size = 80  
num_classes = 10  
epochs = 40 
  
# input image dimensions  
img_rows, img_cols = 28, 28  


# the data, shuffled and split between train and test sets  
(x_train, y_train), (x_test, y_test) = mnist.load_data()     

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)  
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)  
input_shape = (img_rows, img_cols, 1)  
  
x_train = x_train.astype('float32')  
x_test = x_test.astype('float32')  
x_train /= 255  
x_test /= 255  
print('x_train shape:', x_train.shape)  
print(x_train.shape[0], 'train samples')  
print(x_test.shape[0], 'test samples')  
  
# convert class vectors to binary class matrices  
y_train = keras.utils.to_categorical(y_train, num_classes)  
y_test = keras.utils.to_categorical(y_test, num_classes)  
  
model = Sequential()  
model.add(Conv2D(32, kernel_size=(5, 5),  
                 activation='relu',  
                 input_shape=input_shape))  
model.add(Conv2D(32, (5, 5), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Dropout(0.25)) 
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(Conv2D(64, (3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2))) 
model.add(Dropout(0.25))

model.add(Flatten())  
model.add(Dense(512, activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation='softmax'))  
  
model.compile(loss= "categorical_crossentropy",  
              optimizer='adam',  
              metrics=['accuracy'])  

#learning rate annealer
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
		
datagen.fit(x_train)
		
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),  
          epochs=epochs,  
          verbose=2,
		  validation_data=(x_test, y_test),
		  steps_per_epoch=x_train.shape[0]// batch_size,
          callbacks=[learning_rate_reduction])
score = model.evaluate(x_test, y_test, verbose=0)
CENSUS_MODEL=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))+'.h5'
model.save(CENSUS_MODEL)
copy_file_to_gcs(job_dir, CENSUS_MODEL)
print('Test loss:', score[0])  
print('Test accuracy:', score[1])


with file_io.FileIO(os.path.join(job_dir, 'test.csv'), mode='rb') as input_f:
    with file_io.FileIO('test.csv', mode='w+') as output_f:
        output_f.write(input_f.read())
df_x_predict = pd.read_csv('test.csv', header=0)
x_predict=np.array(df_x_predict)
x_predict = x_predict.reshape(x_predict.shape[0], img_rows, img_cols, 1)  
x_predict = x_predict.astype('float32')    
x_predict /= 255 
result=model.predict(x_predict,batch_size=32,verbose=2)
result = np.argmax(result,axis = 1)
result = pd.Series(result,name="Label")
result.to_csv('prediction1.csv',index=True,index_label='ImageId')
copy_file_to_gcs(job_dir, 'prediction1.csv')
