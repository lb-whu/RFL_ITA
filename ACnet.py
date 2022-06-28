import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ops
import numpy as np
class Unet2:
  def __init__(self, name, is_training, ngf=40, norm='instance', num_class = 9,model_path='largefov.npy'):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.num_class = num_class
    self.model_path = model_path
#    self.result_from_contract_layer = {}
  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    self.load_init_model()
    with tf.variable_scope(self.name):
      # conv layers
      c1,c2,c3,layer_down = self.downsample(input,self.init_model)
      print('c1c1c1c1c1',c1.shape)
      print('c2c2c2c2c2',c2.shape)
      print('c3c3c3c3c3c3',c3.shape)
      print('layer_downlayer_downlayer_downlayer_down',layer_down.shape)
      layer_last = self.upsample(c1,c2,c3,layer_down,'layerup_',self.init_model)
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return layer_last
  def downsample(self, input,init_model):
    with tf.variable_scope(self.name):
      with tf.device("/device:GPU:0"):
        c1,layer_1 = ops.layer_down3d2(input,1*self.ngf,wnum = 3,is_training=self.is_training,reuse=self.reuse,
                              keep_prob= 1,init_model = init_model,name = 'conv1')
        c2,layer_2 = ops.layer_down3d(layer_1,2*self.ngf,wnum = 3,is_training=self.is_training,reuse=self.reuse,
                              keep_prob= 1,init_model = init_model,name = 'conv2')
        c3,layer_3 = ops.layer_down3d(layer_2,4*self.ngf,wnum = 3,is_training=self.is_training,reuse=self.reuse,
                              keep_prob= 1,init_model = init_model,name = 'conv3')
        layer_4,_ = ops.layer_down3d(layer_3,8*self.ngf,wnum = 3,is_training=self.is_training,reuse=self.reuse,
                              keep_prob= 1,init_model = init_model,name = 'conv4')
                              
      return c1,c2,c3,layer_4
  
  def upsample(self, c1,c2,c3,input,name,init_model):
    with tf.variable_scope(self.name):
#      nam_tmp = name + 'mid'
#      layer_mid = ops.layer_mid(input,16*self.ngf,8*self.ngf,is_training=self.is_training,reuse=self.reuse,
#                                keep_prob= 1,name = nam_tmp) 
      # deconv layers
      nam_tmp = name + '5'
      with tf.device("/device:GPU:1"):
        layer_5 = ops.layer_up3d(c3,input,4*self.ngf,is_training=self.is_training,
                                  keep_prob= 1,reuse=self.reuse,name = nam_tmp) 
#      with tf.device("/device:GPU:1"):                            
        nam_tmp = name + '6'
        layer_6 = ops.layer_up3d(c2,layer_5,2*self.ngf,is_training=self.is_training,
                                  keep_prob= 1,reuse=self.reuse,name = nam_tmp)  
  
        nam_tmp = name + '7'                      
        layer_7 = ops.layer_up3d2(c1,layer_6,self.ngf,is_training=self.is_training,
                                  keep_prob= 1,reuse=self.reuse,name = nam_tmp)
        nam_tmp = name + '8' 
        layer_last = ops.layer_one(layer_7,4,wnum = 1,is_training=self.is_training,reuse=self.reuse,
                              keep_prob= 1,init_model = init_model,name = nam_tmp)
      print('layer_5',layer_5.shape)
      print('layer_6',layer_6.shape)
      print('layer_7',layer_7.shape)                                               
      print('layer_lastlayer_lastlayer_last::',layer_last.shape) 
      return layer_last
  def upsampleGAN(self,input,name):
    with tf.variable_scope(self.name):
      nam_tmp = name + 'mid'
      layer_mid = ops.layer_mid(input,16*self.ngf,8*self.ngf,is_training=self.is_training,reuse=self.reuse,
                                keep_prob= 1,name = nam_tmp) 

      # deconv layers
      nam_tmp = name + '6'
      layer_6 = ops.layer_upGAN(layer_mid,8*self.ngf,4*self.ngf,is_training=self.is_training,
                                keep_prob= 1,reuse=self.reuse,name = nam_tmp)
      print('layer_6layer_6layer_6::',layer_6.shape)
      nam_tmp = name + '7'
      layer_7 = ops.layer_upGAN(layer_6,4*self.ngf,2*self.ngf,is_training=self.is_training,
                                keep_prob= 1,reuse=self.reuse,name = nam_tmp)
      print('layer_7layer_7layer_7::',layer_7.shape)    
      nam_tmp = name + '8'                      
      layer_8 = ops.layer_upGAN(layer_7,2*self.ngf,self.ngf,is_training=self.is_training,
                                keep_prob= 1,reuse=self.reuse,name = nam_tmp)
      print('layer_8layer_8layer_8::',layer_8.shape)  
      nam_tmp = name + '9'                          
      layer_9 = ops.layer_lastGAN(layer_8,self.ngf,self.num_class,is_training=self.is_training,
                                keep_prob= 1,reuse=self.reuse,name = nam_tmp)
      print('layer_9layer_9layer_9::',layer_9.shape) 
      return layer_9
  def load_init_model(self):
    self.init_model = np.load(self.model_path,encoding="latin1",allow_pickle = True).item()
  def sample(self, input):
    layer_seg = self.__call__(input)
#    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return layer_seg
