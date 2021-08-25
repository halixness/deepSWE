#!/usr/bin/env python
# coding: utf-8

# In[9]:


# In[10]:

'''
partitions = DataPartitions(
    past_frames=4, 
    future_frames=4, 
    root="../datasets/arda/04_21_full/",
    partial = 0.3
)


# In[11]:


dataset = DataGenerator(
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


# In[12]:


X = dataset.get_X()
Y = dataset.get_Y()

X[X > 10e5] = 0 
Y[Y > 10e5] = 0 


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[14]:


sc_img = StandardScaler() # image
sc_vvx = StandardScaler() # vvx
sc_vvy = StandardScaler() # vvy

for sample in X_train:
    for batch in sample:
        for frame in batch:
            sc_img.partial_fit(frame[:,:,0])
            sc_vvx.partial_fit(frame[:,:,1])
            sc_vvy.partial_fit(frame[:,:,1])


# In[15]:


for s, sample in enumerate(X_train):
    for b, batch in enumerate(sample):
        for f, frame in enumerate(batch):
            X_train[s, b, f, :, :, 0] = sc_img.transform(frame[:,:,0])
            X_train[s, b, f, :, :, 1] = sc_vvx.transform(frame[:,:,1])
            X_train[s, b, f, :, :, 2] = sc_vvy.transform(frame[:,:,2])
            
print("X_train ready")        

for s, sample in enumerate(X_test):
    for b, batch in enumerate(sample):
        for f, frame in enumerate(batch):
            X_test[s, b, f, :, :, 0] = sc_img.transform(frame[:,:,0])
            X_test[s, b, f, :, :, 1] = sc_vvx.transform(frame[:,:,1])
            X_test[s, b, f, :, :, 2] = sc_vvy.transform(frame[:,:,2])
            
print("X_test transformed")
            
for s, sample in enumerate(y_train):
    for b, batch in enumerate(sample):
        for f, frame in enumerate(batch):
            y_train[s, b, f, :, :, 0] = sc_img.transform(frame[:,:,0])
  
print("y_train transformed")

for s, sample in enumerate(y_test):
    for b, batch in enumerate(sample):
        for f, frame in enumerate(batch):
            y_test[s, b, f, :, :, 0] = sc_img.transform(frame[:,:,0])
            
print("y_test transformed")


# In[ ]:


del X
del Y


# ### Model

# In[1]:
'''

import numpy as np

import torch as th
import torch.nn as nn
from torchsummary import summary


# In[2]:


class ResNetBlock(nn.Module):
        
    def __init__(self, in_filters, filters, stride):
        super(ResNetBlock, self).__init__()
        
        self.activation = nn.ReLU()
        self.c1 = nn.Conv3d(in_filters, filters, (3, 3, 3), stride, padding=1)
        self.c2 = nn.Conv3d(filters, filters, (3, 3, 3), padding=1)
        self.c3 = nn.Conv3d(in_filters, filters, (1, 1, 1), stride)

    def forward(self, x,):
        
        residual = x
        
        y = self.c1(x)

        # batchnorm here
        y = self.activation(y)
        y = self.c2(y)
        # batchnorm here

        # reshape
        if residual.shape != y.shape:
            residual = self.c3(residual)
            # batchnorm here
            
        return self.activation(residual + y)


# In[4]:


class Encoder(nn.Module):
    
    def __init__(self, channels, past_frames, future_frames):
        super(Encoder, self).__init__()
        

        self.layers = nn.ModuleList([
            nn.Conv3d(channels, 64, (3,3,3), stride=2),
            nn.AvgPool3d((2,2,2)),
            
            ResNetBlock(in_filters = 64, filters = 64, stride = 2),

            nn.Conv3d(64, 128, (1,1,1)),
            ResNetBlock(in_filters = 128, filters = 128, stride = 2),

            nn.Conv3d(128, 256, (1,1,1)),
            ResNetBlock(in_filters = 256, filters = 256, stride = 2),
        ])

        
        '''
        channels = [64, 128, 256, 512]
        blocks = [2, 1, 1, 1]
        layers = []
        
        for i, c in enumerate(channels):

            # transitional block (no skip)
            if i > 0:
                layers.append(nn.Conv3d(channels[i-1], c, (1,3,3), stride=(2,2,2)))
                layers.append(nn.Conv3d(c, c, (3,3,3)))
            
            # resnet block
            for b in blocks:
                layers.append(ResNetBlock(in_filters = c, filters = c, stride = 2))
        
        self.layers = nn.ModuleList(layers)
        '''
        
    
    def forward(self, x):

        for i, l in enumerate(self.layers):
            x = l(x)
            print(x.shape)
        
        return x



# In[5]:


enc = Encoder(3, 4, 4)
summary(enc, (3, 4, 256, 256))

exit()


# In[6]:


enc(th.Tensor(np.random.random((16, 3, 4, 256, 256))))


# In[ ]:



# il decoder satura la memoria
class Decoder(nn.Module):
    
    @nn.compact
    def __call__(self, x):
        
        channels = [128, 64]
        blocks = [3, 3]
        
        for i, c in enumerate(channels):
            
            # stacked blocks
            for b in blocks:
                x = InvResNetBlock(filters=c)(x)
               
            if i is not channels[-1]:
                x = nn.ConvTranspose(c, (3,3,3))(x)
                x = nn.ConvTranspose(c, (3,3,3), strides=(2,2,2))(x)
            
        
        x = nn.ConvTranspose(64, (1,1,1), strides=(2,2,2))(x)
        x = nn.ConvTranspose(64, (7,7,7), strides=(2,2,2))(x)
        x = nn.ConvTranspose(1, (1,1,1))(x)
        
        return x
    
class Autoencoder(nn.Module):
    """Complete ResNet"""
    
    def setup(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def __call__(self, x):
        return self.decoder(self.encoder(x))


# In[ ]:





# In[ ]:





# In[ ]:




