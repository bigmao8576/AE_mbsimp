
import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv1D,Flatten,MaxPool1D,AveragePooling1D,UpSampling1D,BatchNormalization,Conv1DTranspose
from tensorflow.keras import Model



class cnn_encoder(Model):
  def __init__(self,filter_num, ch):
    super(cnn_encoder, self).__init__()
    
    self.output_ch = ch

      
    self.c1 = Conv1D(filter_num, 5, strides=2,padding='valid')
    
    self.d1 = Dense(filter_num)
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
  def __init__(self,filter_num,ch):
    super(cnn_decoder, self).__init__()
    
    self.output_ch = ch

    self.up = UpSampling1D()
    self.up_mask = UpSampling1D()
    
    
    self.c1 = Conv1D(filter_num, 3, strides=1,padding='valid')
    self.c2 = Conv1D(filter_num, 3, strides=1,padding='valid')

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

class ae_hie(Model):
    def __init__(self):
        super(ae_hie, self).__init__()
        
        self.cen_L1 = cnn_encoder(32,7)
        self.cen_L1._name = 'cen_l1'
        
        self.cen_L2 = cnn_encoder(64,13)
        self.cen_L2._name = 'cen_l2'
        
        self.cen_L3 = cnn_encoder(128,25)#0.2148
        self.cen_L3._name = 'cen_l3'
        
        self.cen_L4 = cnn_encoder(256,49)
        self.cen_L4._name = 'cen_l4'
        
        self.cen_L5 = cnn_encoder(512,97)
        self.cen_L5._name = 'cen_l5'
        
        self.cen_L6 = cnn_encoder(1024,193)
        self.cen_L6._name = 'cen_l6'
        
        self.cen_L7 = cnn_encoder(2048,385)
        self.cen_L7._name = 'cen_l7'
        
        
        self.cde_L1 = cnn_decoder(16,4)
        self.cde_L1._name = 'cde_l1'
        
        self.cde_L2 = cnn_decoder(32,7)
        self.cde_L2._name = 'cde_l2'
        
        self.cde_L3 = cnn_decoder(64,13)
        self.cde_L3._name = 'cde_l3'
        
        self.cde_L4 = cnn_decoder(128,25)
        self.cde_L4._name = 'cde_l4'
        
        self.cde_L5 = cnn_decoder(256,49)
        self.cde_L5._name = 'cde_l5'
        
        self.cde_L6 = cnn_decoder(512,97)
        self.cde_L6._name = 'cde_l6'
        
        self.cde_L7 = cnn_decoder(1024,193)
        self.cde_L7._name = 'cde_l7'
        
        self.dic = {}
    
    def call(self, x_input):
        
        
        latent_L2 = self.cen_L1(x_input) # [b, 9344, 4] => [b, 4672, 7]
        latent_L3 = self.cen_L2(latent_L2) # [b, 4672, 7] => [b, 2336, 13] 
        latent_L4 = self.cen_L3(latent_L3) # [b, 2336, 13] => [b, 1168, 25]
        latent_L5 = self.cen_L4(latent_L4) # [b, 1168, 25] => [b, 584, 49] 
        latent_L6 = self.cen_L5(latent_L5) # [b, 584, 49] => [b, 292, 97]
        latent_L7 = self.cen_L6(latent_L6) # [b, 292, 97] => [b, 146, 193]
        latent = self.cen_L7(latent_L7) #[b, 146, 193] => [b, 73, 385]
        
        L7_rec = self.cde_L7(latent) # [b, 73, 385] => [b, 146, 193]
        L6_rec = self.cde_L6(L7_rec) # [b, 146, 193] => [b, 292, 97]
        L5_rec = self.cde_L5(L6_rec) # [b, 292, 97] => [b, 584, 49]
        L4_rec = self.cde_L4(L5_rec) # [b, 584, 49] => [b, 1168, 25]
        L3_rec = self.cde_L3(L4_rec) # [b, 1168, 25] => [b, 2336, 13]
        L2_rec = self.cde_L2(L3_rec) # [b, 2336, 13] => [b, 4672, 7]
        L1_rec = self.cde_L1(L2_rec) # [b, 4672, 7] => [b, 9344, 4]
        
        L1_rec_int = self.cde_L1(latent_L2) # [b, 4672, 7] => [b, 9344, 4], paired with x_input
        L2_rec_int = self.cde_L2(latent_L3) # [b, 2336, 13] => [b, 4672, 7], paired with latent_L2
        L3_rec_int = self.cde_L3(latent_L4) # [b, 1168, 25] => [b, 2336, 13], paired with latent_L3
        L4_rec_int = self.cde_L4(latent_L5) # [b, 584, 49] => [b, 1168, 25], paired with latent_L4
        L5_rec_int = self.cde_L5(latent_L6) # [b, 292, 97] => [b, 584, 49], paired with latent_L5
        L6_rec_int = self.cde_L6(latent_L7) # [b, 146, 193] => [b, 292, 97], paired with latent_L6
        L7_rec_int = self.cde_L7(latent) # [b, 73, 385] => [b, 146, 193], paired with latent_L7

        
        
        
        
        self.dic['la_l1'] = x_input
        self.dic['la_l2'] = latent_L2
        self.dic['la_l3'] = latent_L3
        self.dic['la_l4'] = latent_L4
        self.dic['la_l5'] = latent_L5
        self.dic['la_l6'] = latent_L6
        self.dic['la_l7'] = latent_L7
        self.dic['la'] = latent
                
        self.dic['l7_rec'] = L7_rec
        self.dic['l6_rec'] = L6_rec
        self.dic['l5_rec'] = L5_rec
        self.dic['l4_rec'] = L4_rec
        self.dic['l3_rec'] = L3_rec
        self.dic['l2_rec'] = L2_rec
        self.dic['l1_rec'] = L1_rec
        
        self.dic['l1_int'] = L1_rec_int
        self.dic['l2_int'] = L2_rec_int
        self.dic['l3_int'] = L3_rec_int
        self.dic['l4_int'] = L4_rec_int
        self.dic['l5_int'] = L5_rec_int
        self.dic['l6_int'] = L6_rec_int
        self.dic['l7_int'] = L7_rec_int
        
        
        return self.dic
    def decoder(self, latent):
        latent_L7_rec = self.cde_L7(latent) # [b, 73, 385] => [b, 146, 193]
        latent_L6_rec = self.cde_L6(latent_L7_rec) # [b, 146, 193] => [b, 292, 97]
        latent_L5_rec = self.cde_L5(latent_L6_rec) # [b, 292, 97] => [b, 584, 49]
        latent_L4_rec = self.cde_L4(latent_L5_rec) # [b, 584, 49] => [b, 1168, 25]
        latent_L3_rec = self.cde_L3(latent_L4_rec) # [b, 1168, 25] => [b, 2336, 13]
        latent_L2_rec = self.cde_L2(latent_L3_rec) # [b, 2336, 13] => [b, 4672, 7]
        latent_L1_rec = self.cde_L1(latent_L2_rec) # [b, 4672, 7] => [b, 9344, 4]

        return latent_L1_rec