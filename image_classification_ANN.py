#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# In[2]:


#Intialising the CNN
classifier=Sequential()
#step1 - Convolution
classifier.add(Conv2D(32,(3,3), input_shape=(64, 64, 3),activation='relu'))

#step2- Adding a second Convolutional layer
classifier.add(Conv2D(32,(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


# In[3]:


#step3- Flattening
classifier.add(Flatten())

#step4 -Full Connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))


# In[4]:


# step5- Compiling the CNN
classifier.compile(optimizer ='adam', loss='binary_crossentropy',metrics=['accuracy'])


# In[5]:


#Part2- Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)


# In[6]:


training_set=train_datagen.flow_from_directory('E:/Placement courses/placement course projects/Python/dataset_cat_dogs/training_set',
                                               target_size=(64,64),batch_size=32,class_mode='binary')


# In[7]:


test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('E:/Placement courses/placement course projects/Python/dataset_cat_dogs/test_set',target_size=(64,64),batch_size=32,class_mode='binary')


# In[8]:


classifier.fit(training_set,
                        steps_per_epoch=4000,
                        epochs=10,validation_data=test_set,
                        validation_steps=10)


# In[9]:


import numpy as np
from keras .preprocessing import image
test_image=image.load_img('E:/Placement courses/placement course projects/Python/dataset_cat_dogs/cat4029.jpg',
                         target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'
    
print(prediction)


# In[ ]:




