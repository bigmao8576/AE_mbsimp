'''

only the output of the last level is expected to be the same with the input signals.
'''



import tensorflow as tf
import os

import utils
import numpy as np
import matplotlib.pyplot as plt


from scipy import io
from scipy import interpolate

import pickle

from tensorflow.keras.layers import Dense, Conv1D,Flatten,MaxPool1D,AveragePooling1D,UpSampling1D,BatchNormalization,Conv1DTranspose
from tensorflow.keras import Model



signal_pool, mask_pool = pickle.load( open( "signal_pool.pkl", "rb" ) )

signal_pool = np.concatenate([signal_pool, mask_pool],-1)


batch_size=64
r_data = tf.data.Dataset.from_tensor_slices((signal_pool))
r_data = r_data.shuffle(10000)
r_data = r_data.batch(batch_size)

temp_iter = r_data.__iter__()
x_signal = temp_iter.__next__()


class cnn_encoder(Model):
  def __init__(self,ch):
    super(cnn_encoder, self).__init__()
    
    self.output_ch = ch
    
    self.c1 = Conv1D(32, 5, strides=2,padding='valid')
    
    self.d1 = Dense(32)
    self.d2 = Dense(self.output_ch)

    self.m_mask = MaxPool1D(pool_size=2, strides=2)

  def call(self, x_input):
      
    input_ch = x_input.shape[-1]
      
    mask_part = tf.slice(x_input, [0, 0, input_ch-1], [-1, -1, -1])
    mask_part = self.m_mask(mask_part)
    
    
    
    x = tf.slice(x_input, [0, 0, 0], [-1, -1, input_ch-1])

    x =tf.pad(x, tf.constant([[0, 0,],[2, 2,], [0, 0]]), "SYMMETRIC") 
    
    x = self.c1(x)
    x = tf.nn.leaky_relu(x)
    
    x = self.d1(x)
    x = tf.nn.leaky_relu(x)
    
    x = self.d2(x)
    
    assert x.shape[1] == mask_part.shape[1]
    
    mask_all = tf.tile(mask_part, [1,1,self.output_ch])
    
    x = tf.math.multiply(x,mask_all)
    
    x = tf.concat([x,mask_part],-1)
    
    return x

class cnn_decoder(Model):
  def __init__(self,ch):
    super(cnn_decoder, self).__init__()
    
    self.output_ch = ch

    self.up = UpSampling1D()
    self.up_mask = UpSampling1D()
    
    self.c1 = Conv1D(32, 3, strides=1,padding='valid')
    self.c2 = Conv1D(32, 3, strides=1,padding='valid')

    self.d1 = Dense(ch)



  def call(self, x_input):
      
    input_ch = x_input.shape[-1]
    
    mask_part = tf.slice(x_input, [0, 0, input_ch-1], [-1, -1, -1])
    mask_part = self.up_mask(mask_part)
    
    x = tf.slice(x_input, [0, 0, 0], [-1, -1, input_ch-1])


    x = tf.pad(x, tf.constant([[0, 0], [1, 1],[0, 0]]), "SYMMETRIC")
    x = self.c1(x)
    x = tf.nn.leaky_relu(x)
    
    x = self.up(x)
    
    x = tf.pad(x, tf.constant([[0, 0], [1, 1],[0, 0]]), "SYMMETRIC")
    x = self.c2(x)
    x = tf.nn.leaky_relu(x)
    
    assert x.shape[1] == mask_part.shape[1]
    
    mask_all = tf.tile(mask_part, [1,1,self.output_ch])
        
    x = self.d1(x)
    
    x = tf.math.multiply(x,mask_all)
    
    x = tf.concat([x,mask_part],-1)

    return x

class cnn_encoder_hie(Model):
    def __init__(self):
        super(cnn_encoder_hie, self).__init__()
        
        self.cen_L1 = cnn_encoder(7)
        self.cen_L2 = cnn_encoder(13)
        self.cen_L3 = cnn_encoder(25)
        self.cen_L4 = cnn_encoder(49)
        self.cen_L5 = cnn_encoder(97)
        self.cen_L6 = cnn_encoder(193)
        self.cen_L7 = cnn_encoder(385)
    
    def call(self, x_input):

        latent_L1 = self.cen_L1(x_input)
        latent_L2 = self.cen_L2(latent_L1)
        latent_L3 = self.cen_L3(latent_L2)
        latent_L4 = self.cen_L4(latent_L3)
        latent_L5 = self.cen_L5(latent_L4)
        latent_L6 = self.cen_L6(latent_L5)
        latent_L7 = self.cen_L7(latent_L6)
        
        return latent_L7

class cnn_decoder_hie(Model):
    def __init__(self):
        super(cnn_decoder_hie, self).__init__()

        self.cde_L1 = cnn_decoder(4)
        self.cde_L2 = cnn_decoder(7)
        self.cde_L3 = cnn_decoder(13)
        self.cde_L4 = cnn_decoder(25)
        self.cde_L5 = cnn_decoder(49)
        self.cde_L6 = cnn_decoder(97)
        self.cde_L7 = cnn_decoder(193)

    def call(self, x_input):
        latent_L7_rec = self.cde_L7(x_input)
        latent_L6_rec = self.cde_L6(latent_L7_rec)
        latent_L5_rec = self.cde_L5(latent_L6_rec)
        latent_L4_rec = self.cde_L4(latent_L5_rec)
        latent_L3_rec = self.cde_L3(latent_L4_rec)
        latent_L2_rec = self.cde_L2(latent_L3_rec)
        latent_L1_rec = self.cde_L1(latent_L2_rec)
        
        return latent_L1_rec


cen_hie = cnn_encoder_hie()
cde_hie = cnn_decoder_hie()





inputs = tf.keras.Input(shape=(signal_pool.shape[1], signal_pool.shape[2]))

lat = cen_hie(inputs)
x_rec= cde_hie(lat)
auto_hie = tf.keras.Model(inputs=inputs, outputs=x_rec)
optimizer = tf.keras.optimizers.Adam(1e-5 )


def dis_loss_pair(x_in,x_out):
    x_ch = x_in.shape[-1]
    mask = tf.slice(x_in, [0, 0, x_ch-1], [-1, -1, -1])
    x_sig = tf.slice(x_in, [0, 0, 0], [-1, -1, x_ch-1])
    

    x_out_sig = tf.slice(x_out, [0, 0, 0], [-1, -1, x_ch-1])   
    
    d_loss = tf.reduce_sum((x_sig-x_out_sig)**2)/(tf.reduce_sum(mask)*(x_ch-1))
    
    return d_loss

def dis_loss(x,la=0.01):
    
    
    z = cen_hie(x)
    l1_rec= cde_hie(z)
    
    loss_L1 = dis_loss_pair(x,l1_rec)
       
    z_ch = z.shape[-1]
    z_mask = tf.slice(z, [0, 0, z_ch-1], [-1, -1, -1])
    z_sig = tf.slice(z, [0, 0, 0], [-1, -1, z_ch-1])
    
    p_loss = tf.reduce_sum(z_sig**2)/(tf.reduce_sum(z_mask)*(z_ch-1))
    
    total_loss = loss_L1+la*p_loss
    
    return total_loss


@tf.function
def train_step(x):
   
    with tf.GradientTape() as tape:
        diff_loss = dis_loss(x)
    
    grad= tape.gradient(diff_loss, auto_hie.trainable_variables)
    optimizer.apply_gradients(zip(grad, auto_hie.trainable_variables))
    
    

total_loss = []


for epoch in range(10000):
    ep_loss = []

    
    for x_data in r_data: 
        train_step(x_data)    
        diff_loss = dis_loss(x_data)
        
        ep_loss.append(diff_loss.numpy())
        
    
    
    if epoch%1==0:   
        print(epoch,'_____',np.mean(ep_loss))
        total_loss.append(np.mean(ep_loss))

        raw_data = signal_pool[:10,:,:]
        l1_z = cen_hie(raw_data)
        l1_re = cde_hie(l1_z)

        
        plt.plot(np.log10(total_loss));plt.show()
        plt.plot(raw_data[1,:,2]);plt.plot(l1_re[1,:,2]);plt.show()
        plt.plot(raw_data[1,:100,2]);plt.plot(l1_re[1,:100,2]);plt.show()
   

# =============================================================================
# 
# cen_L1 = cnn_encoder(7)
# cen_L2 = cnn_encoder(13)
# cen_L3 = cnn_encoder(25)
# cen_L4 = cnn_encoder(49)
# cen_L5 = cnn_encoder(97)
# cen_L6 = cnn_encoder(193)
# cen_L7 = cnn_encoder(385)
# 
# cde_L1 = cnn_decoder(4)
# cde_L2 = cnn_decoder(13)
# cde_L3 = cnn_decoder(25)
# cde_L4 = cnn_decoder(49)
# cde_L5 = cnn_decoder(97)
# cde_L6 = cnn_decoder(193)
# cde_L7 = cnn_decoder(385)
# 
# 
# latent_L1 = cen_L1(x_signal)
# latent_L2 = cen_L2(latent_L1)
# latent_L3 = cen_L3(latent_L2)
# latent_L4 = cen_L4(latent_L3)
# latent_L5 = cen_L5(latent_L4)
# latent_L6 = cen_L6(latent_L5)
# latent_L7 = cen_L7(latent_L6)
# 
# latent_L7_rec = cde_L7(latent_L7)
# latent_L6_rec = cde_L6(latent_L7_rec)
# latent_L5_rec = cde_L5(latent_L6_rec)
# latent_L4_rec = cde_L4(latent_L5_rec)
# latent_L3_rec = cde_L3(latent_L4_rec)
# latent_L2_rec = cde_L2(latent_L3_rec)
# latent_L1_rec = cde_L1(latent_L2_rec)
# 
# =============================================================================
