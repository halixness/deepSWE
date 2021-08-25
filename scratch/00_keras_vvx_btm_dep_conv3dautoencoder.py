#!/usr/bin/env python
# coding: utf-8

# ### Drive Mount

# In[ ]:


from utils.data_legacy.dataloader import DataPartitions, DataGenerator


# In[ ]:


partitions = DataPartitions(
    past_frames=10, 
    future_frames=4, 
    root="../datasets/arda/04_21_full/", 
    train_size=0.9
)


# In[ ]:


train_dataset = DataGenerator(
    root="../datasets/arda/04_21_full/", 
    filenames=partitions.get_areas(), 
    dataset_partitions=partitions.get_train(), 
    past_frames=partitions.past_frames, 
    future_frames=partitions.future_frames, 
    input_dim=(partitions.past_frames, 256, 256, 3),  
    output_dim=(partitions.future_frames, 256, 256, 1), 
    batch_size=16, 
    n_channels=1, 
    shuffle=True,
    buffer_size = 1e3,
    buffer_memory = 100
)


# In[ ]:


test_dataset = DataGenerator(
    root="../datasets/arda/04_21_full/", 
    filenames=partitions.get_areas(), 
    dataset_partitions=partitions.get_test(), 
    past_frames=partitions.past_frames, 
    future_frames=partitions.future_frames, 
    input_dim=(partitions.past_frames, 256, 256, 3),  
    output_dim=(partitions.future_frames, 256, 256, 1), 
    batch_size=16, 
    n_channels=1, 
    shuffle=True,
    buffer_size = 1e3,
    buffer_memory = 100
)


# <hr>
# 

# In[ ]:


import numpy as np
import os
import tqdm
import matplotlib.pyplot as plt
import tqdm

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import keras.layers.advanced_activations as advanced_activations
import keras.layers as layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv3D, Conv3DTranspose
from tensorflow.keras.layers import AveragePooling3D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.models import Model


# In[ ]:


if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

else:

   print("Please install GPU version of TF")


# In[9]:


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        
    def loss_plot(self):
        plt.plot(range(len(self.losses)), self.losses)
        
    def accuracy_plot(self):
        plt.plot(range(len(self.accuracy)), self.accuracy)

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))

        #print("Loss: {}, Val_loss: {}, Accuracy: {}".format(logs.get('loss'), logs.get('loss'), logs.get('accuracy')))


# In[10]:


tf.__version__


# ### Model

# In[12]:


# In[13]:


# (batch_size, channels, depth, height, width).
sample_shape = (10, 256, 256, 3)
inputs = Input(shape = sample_shape)

x = inputs

x = Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape)(x)
x = BatchNormalization()(x)
x = AveragePooling3D(pool_size=(2, 2, 2))(x)

x = Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_uniform')(x)
x = BatchNormalization()(x)
x = AveragePooling3D(pool_size=(2, 2, 2))(x)

x = Conv3DTranspose(filters=64, kernel_size=(3,3,3), strides=(2,2,2), padding="same",activation='relu')(x)
x = BatchNormalization()(x)

x = Conv3DTranspose(filters=32, kernel_size=(3,3,3), strides=(1,2,2), padding="same",activation='relu')(x)
x = BatchNormalization()(x)

x = Conv3DTranspose(filters=1, kernel_size=(3,3,3), strides=(1,1,1), padding="same",activation='relu')(x)
x = BatchNormalization()(x)

model = Model(inputs, x)
model.summary()


# In[14]:


#model.compile(optimizer = Adam(learning_rate = 1e-3), loss = "binary_crossentropy", metrics = ["accuracy"])
model.compile(optimizer = Adam(learning_rate = 1e-3), loss = "mean_absolute_error", metrics = ["accuracy"])


# In[15]:


history = LossHistory()


# In[ ]:


model.fit(
    x = train_dataset,
    validation_data = test_dataset,
    use_multiprocessing = True,
    workers = 6,
    epochs = 1,
    callbacks=[history]
)


# In[ ]:


predictions = model.predict(train_d.__getitem__(ind)[0])[0]
true = train_d.__getitem__(ind)[1][0]


# In[ ]:


fig, axs = plt.subplots(1,4, figsize=(15,5))

for i, frame in enumerate(predictions):
    axs[i].matshow(frame.reshape(256, 256))


# In[ ]:


fig, axs = plt.subplots(1,4, figsize=(15,5))

for i, frame in enumerate(true):
    axs[i].matshow(frame.reshape(256, 256))


# In[ ]:


plt.plot(range(4), [np.mean(x) for x in predictions])
plt.plot(range(4), [np.mean(x) for x in true])


# In[ ]:


ind = 6

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
fig.suptitle('frame 1 - frame 11 pred - frame 11 true')

ax1.matshow(train_d.__getitem__(ind)[0][0][0].reshape(256,256))

ax2.matshow(model.predict(train_d.__getitem__(ind)[0])[0][0].reshape(256,256))

ax3.matshow(train_d.__getitem__(ind)[1][0][0].reshape(256,256))


# In[ ]:


f = np.concatenate((train_d.__getitem__(ind)[0][0], train_d.__getitem__(ind)[1][0]))
plt.plot(range(14), [np.mean(x) for x in f])

plt.plot(range(10, 14), [np.mean(x) for x in model.predict(train_d.__getitem__(ind)[0])[0]])


# In[ ]:


history.accuracy_plot()


# In[ ]:


history.loss_plot()


# In[ ]:




