import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
import numpy as np
## Layers: follow the naming convention used in the original paper
### Generator layers
def layer_one(input, k,wnum, reuse=False, keep_prob= 1, is_training=True, init_model=None,name='layer_1st'):
  """ 2 3x3 Convolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    num = input.get_shape()[1]
    #conv1
    weights_1 = _weights("weights_1",
      shape=[wnum, wnum, wnum, input.get_shape()[4], k],init = None)
    biases_1 = _biases("biases_1", [k],init = None)
    result_conv_1 = tf.nn.conv3d(input, weights_1,
				                              strides=[1, 1, 1, 1, 1], padding='VALID', name='conv_1')
#    result_relu_1 = tf.nn.sigmoid(result_conv_1, name='soft_max')
    result_relu_1 = tf.nn.softmax(result_conv_1, name='soft_max') 
#    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), is_training, 'batch',name = 'conv1')
#    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')   
    return result_relu_1
def layer_down3d(input, k,wnum, reuse=False, keep_prob= 1, is_training=True, init_model=None,name='layer_down'):
  """ 2 3x3 Convolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    num = input.get_shape()[1]
    #conv1
    weights_1 = _weights("weights_1",
      shape=[3, 3, 3, input.get_shape()[4], int(k/2)],init = None)
    biases_1 = _biases("biases_1", [int(k/2)],init = None)
    result_conv_1 = tf.nn.conv3d(input, weights_1,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), is_training, 'batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",
      shape=[3, 3, 3, result_relu_1.get_shape()[4], k],init = None)
    biases_2 = _biases("biases_2", [k],init = None)
    result_conv_2 = tf.nn.conv3d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_training, 'batch',name = 'conv2')
    
    #resnet cov
    weights_111 = _weights("weights_111",
      shape=[1, 1, 1, input.get_shape()[4], k])
    res_input = tf.nn.conv3d(input, weights_111,strides=[1, 1, 1, 1, 1],
                                      padding='SAME', name='conv_11')
    res_input = _norm(res_input, is_training, 'batch',name = 'norm3')                                                                                                                     
    resnormed = tf.add(normalized2,res_input)
    result_relu_2 = tf.nn.relu(resnormed, name='relu_2')
    result_maxpool2 = tf.nn.max_pool3d(input=result_relu_2, ksize=[1, 2, 2, 2, 1],
				                  strides=[1, 2, 2, 2, 1], padding='VALID', name='maxpool')
      
    return result_relu_2,result_maxpool2
    
def layer_down3d2(input, k,wnum, reuse=False, keep_prob= 1, is_training=True, init_model=None,name='layer_down'):
  """ 2 3x3 Convolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    num = input.get_shape()[1]
    #conv1
    weights_1 = _weights("weights_1",
      shape=[3, 3, 3, input.get_shape()[4], int(k/2)],init = None)
    biases_1 = _biases("biases_1", [int(k/2)],init = None)
    result_conv_1 = tf.nn.conv3d(input, weights_1,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), is_training, 'batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
    #conv2
    weights_2 = _weights("weights_2",
      shape=[3, 3, 3, result_relu_1.get_shape()[4], k],init = None)
    biases_2 = _biases("biases_2", [k],init = None)
    result_conv_2 = tf.nn.conv3d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_training, 'batch',name = 'conv2')
    
    result_relu_2 = tf.nn.relu(normalized2, name='relu_2')
    result_maxpool2 = tf.nn.max_pool3d(input=result_relu_2, ksize=[1, 2, 2, 2, 1],
				                  strides=[1, 2, 2, 2, 1], padding='VALID', name='maxpool')
      
#    with tf.device("/device:GPU:1"):  
      #resnet cov
#      weights_111 = _weights("weights_111",
#        shape=[1, 1, 1, input.get_shape()[4], k])
#      res_input = tf.nn.conv3d(input, weights_111,strides=[1, 1, 1, 1, 1],
#                                        padding='SAME', name='conv_11')
#      res_input = _norm(res_input, is_training, 'batch',name = 'norm3')                                                                                                                     
#      resnormed = tf.add(normalized2,res_input)
#      result_relu_2 = tf.nn.relu(resnormed, name='relu_2')
#      result_maxpool2 = tf.nn.max_pool3d(input=result_relu_2, ksize=[1, 2, 2, 2, 1],
#  				                  strides=[1, 2, 2, 2, 1], padding='VALID', name='maxpool')
      
    return result_relu_2,result_maxpool2
def layer_down3d3(input, k,wnum, reuse=False, keep_prob= 1, is_training=True, init_model=None,name='layer_down'):
  """ 2 3x3 Convolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    num = input.get_shape()[1]
    #conv1
    weights_1 = _weights("weights_1",
      shape=[3, 3, 3, input.get_shape()[4], k],init = None)
    biases_1 = _biases("biases_1", [k],init = None)
    result_conv_1 = tf.nn.conv3d(input, weights_1,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), is_training, 'batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",
      shape=[3, 3, 3, result_relu_1.get_shape()[4], k],init = None)
    biases_2 = _biases("biases_2", [k],init = None)
    result_conv_2 = tf.nn.conv3d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_training, 'batch',name = 'conv2')
    
    #resnet cov
    weights_111 = _weights("weights_111",
      shape=[3, 3, 3, input.get_shape()[4], k])
    res_input = tf.nn.conv3d(input, weights_111,strides=[1, 1, 1, 1, 1],dilations = [1, int(k/16), int(k/16), int(k/16), 1],
                                      padding='VALID', name='conv_11')
    sp1 = input.get_shape()[1] - res_input.get_shape()[1]
    sp2 = input.get_shape()[2] - res_input.get_shape()[2]
    sp3 = input.get_shape()[3] - res_input.get_shape()[3]
    paddings = [[0,0],[int(sp1//2),int(sp1//2)],[int(sp2//2),int(sp2//2)],[int(sp3//2),int(sp3//2)],[0,0]]
    res_input = tf.pad(tensor=res_input, paddings=paddings, mode="REFLECT")
                                                                                                                         
    resnormed = tf.add(normalized2,res_input)
    result_relu_2 = tf.nn.relu(resnormed, name='relu_2')   
    return result_relu_2



def layer_mid(input, k_down,k_up,reuse=False, keep_prob= 1, is_training=True, name='layer_mid'):
  with tf.variable_scope(name, reuse=reuse):

    
#    result_merge = tf.squeeze(tf.stack([f1,input,f2],1),0)
#    print('layer_mid    result_merge:',result_merge.shape)
    #conv1
    print('input dtype~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~::',input.dtype)
    result = upsample(input, 2, result_dropout.get_shape()[4]) 
    return result
    

def layer_up3d(result_from_contract_layer,result_from_upsampling, k_up,reuse=False, keep_prob= 1, is_training=True, name='layer_up'):
  """ 2 3x3 deConvolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    result_up = upsample(result_from_upsampling, 2, result_from_upsampling.get_shape()[4])    
    result_merge = copy_and_crop_and_merge(result_from_contract_layer, result_up)
    '''
    w_1 = tf.reduce_mean(tf.reduce_mean(result_merge, axis=1, keep_dims=True),axis=2, keep_dims=True)
    weights_11 = _weights("weights_11",shape=[1, 1, result_merge.get_shape()[3],result_merge.get_shape()[3]],init = None)
    w_1 = tf.nn.conv2d(w_1, weights_11,strides=[1, 1, 1, 1], padding='SAME', name='conv_11')
    
    w_2 = tf.reduce_max(tf.reduce_max(result_merge, axis=1, keep_dims=True),axis=2, keep_dims=True)                                                                               
    weights_12 = _weights("weights_12",shape=[1, 1, result_merge.get_shape()[3],result_merge.get_shape()[3]],init = None)
    w_2 = tf.nn.conv2d(w_2, weights_12,strides=[1, 1, 1, 1], padding='SAME', name='conv_12')    
    w = w_1+w_2    
    w_norm = _norm(w, is_training, 'batch',name = 'w_norm')
    w = tf.nn.sigmoid(w_norm,name = 'sig')
    result = tf.multiply(result_merge, w) 
    '''
    #conv1
    print('result_merge:',result_merge.shape)
    weights_1 = _weights("weights_1",
      shape=[3, 3, 3, result_merge.get_shape()[4], k_up])
    biases_1 = _biases("biases_1", [k_up])
    result_conv_1 = tf.nn.conv3d(result_merge, weights_1,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), is_training, 'batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",
      shape=[3, 3, 3, result_relu_1.get_shape()[4], k_up])
    biases_2 = _biases("biases_2", [k_up])
    result_conv_2 = tf.nn.conv3d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_training, 'batch',name = 'conv2')
    
    #resnet cov
    weights_11 = _weights("weights_res",
      shape=[1, 1, 1, result_merge.get_shape()[4], k_up])
    res_input = tf.nn.conv3d(result_merge, weights_11,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_res') 
    res_input = _norm(res_input, is_training, 'batch',name = 'norm3')                                                                                  
    resnormed = tf.add(normalized2,res_input)
    
    result_relu_2 = tf.nn.relu(resnormed, name='relu_2')
    result_dropout = tf.nn.dropout(x=result_relu_2, keep_prob=keep_prob)
    
    # unsample
#    result = tf.image.resize_bilinear(result_dropout,[result_dropout.get_shape()[1]*2,result_dropout.get_shape()[2]*2,result_dropout.get_shape()[3]*2])
    
    '''                        
    weights_3 = _weights("weights_3",
      shape=[2, 2, k_up, result_relu_2.get_shape()[3]])
    biases_3 = _biases("biases_3", [k_up])
    output_size = result_relu_2.get_shape()[1]*2
    output_shape = [result_relu_2.get_shape()[0], output_size, output_size, k_up]
    result_up_conv = tf.nn.conv2d_transpose(result_relu_2, weights_3,output_shape=output_shape,
                                      strides=[1, 2, 2, 1], padding='VALID',name = 'Up_Sample')
    result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up_conv, biases_3, name='add_bias'), name='relu_3')
    '''
		# dropout
    return result_dropout
def layer_up3d2(result_from_contract_layer,result_from_upsampling, k_up,reuse=False, keep_prob= 1, is_training=True, name='layer_up'):
  """ 2 3x3 deConvolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    result_up = upsample(result_from_upsampling, 2, result_from_upsampling.get_shape()[4])    
    result_merge = copy_and_crop_and_merge(result_from_contract_layer, result_up)
    '''
    w_1 = tf.reduce_mean(tf.reduce_mean(result_merge, axis=1, keep_dims=True),axis=2, keep_dims=True)
    weights_11 = _weights("weights_11",shape=[1, 1, result_merge.get_shape()[3],result_merge.get_shape()[3]],init = None)
    w_1 = tf.nn.conv2d(w_1, weights_11,strides=[1, 1, 1, 1], padding='SAME', name='conv_11')
    
    w_2 = tf.reduce_max(tf.reduce_max(result_merge, axis=1, keep_dims=True),axis=2, keep_dims=True)                                                                               
    weights_12 = _weights("weights_12",shape=[1, 1, result_merge.get_shape()[3],result_merge.get_shape()[3]],init = None)
    w_2 = tf.nn.conv2d(w_2, weights_12,strides=[1, 1, 1, 1], padding='SAME', name='conv_12')    
    w = w_1+w_2    
    w_norm = _norm(w, is_training, 'batch',name = 'w_norm')
    w = tf.nn.sigmoid(w_norm,name = 'sig')
    result = tf.multiply(result_merge, w) 
    '''
    #conv1

    print('result_merge:',result_merge.shape)
    weights_1 = _weights("weights_1",
      shape=[3, 3, 3, result_merge.get_shape()[4], k_up])
    biases_1 = _biases("biases_1", [k_up])
    result_conv_1 = tf.nn.conv3d(result_merge, weights_1,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), is_training, 'batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
      
    #conv2

    weights_2 = _weights("weights_2",
      shape=[3, 3, 3, result_relu_1.get_shape()[4], k_up])
    biases_2 = _biases("biases_2", [k_up])
    result_conv_2 = tf.nn.conv3d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_training, 'batch',name = 'conv2')
    result_relu_2 = tf.nn.relu(normalized2, name='relu_2')
    result_dropout = tf.nn.dropout(x=result_relu_2, keep_prob=keep_prob)

      
#      #resnet cov
#      weights_11 = _weights("weights_res",
 #       shape=[1, 1, 1, result_merge.get_shape()[4], k_up])
#      res_input = tf.nn.conv3d(result_merge, weights_11,
#  				                              strides=[1, 1, 1, 1, 1], padding='SAME', name='conv_res') 
#      res_input = _norm(res_input, is_training, 'batch',name = 'norm3')                                                                                  
#      resnormed = tf.add(normalized2,res_input)
      
#      result_relu_2 = tf.nn.relu(resnormed, name='relu_2')
#      result_dropout = tf.nn.dropout(x=result_relu_2, keep_prob=keep_prob)
    
    # unsample
#    result = tf.image.resize_bilinear(result_dropout,[result_dropout.get_shape()[1]*2,result_dropout.get_shape()[2]*2,result_dropout.get_shape()[3]*2])
    
    '''                        
    weights_3 = _weights("weights_3",
      shape=[2, 2, k_up, result_relu_2.get_shape()[3]])
    biases_3 = _biases("biases_3", [k_up])
    output_size = result_relu_2.get_shape()[1]*2
    output_shape = [result_relu_2.get_shape()[0], output_size, output_size, k_up]
    result_up_conv = tf.nn.conv2d_transpose(result_relu_2, weights_3,output_shape=output_shape,
                                      strides=[1, 2, 2, 1], padding='VALID',name = 'Up_Sample')
    result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up_conv, biases_3, name='add_bias'), name='relu_3')
    '''
		# dropout
    return result_dropout
def upsample(input, upsamplescale, channel_count):
    weights_up = _weights("weights_up",
      shape=[2, 2, 2, channel_count, channel_count])
    deconv = tf.nn.conv3d_transpose(value=input, filter=weights_up, 
                        output_shape=[input.get_shape()[0], input.get_shape()[1]*upsamplescale, input.get_shape()[2]*upsamplescale, input.get_shape()[3]*upsamplescale, channel_count],
                                strides=[1, upsamplescale, upsamplescale, upsamplescale, 1],
                                padding="SAME", name='UpsampleDeconv')
    return deconv
                 
def layer_upGAN(result_merge, k_down,k_up,reuse=False, keep_prob= 1, is_training=True, name='layer_up'):
  """ 2 3x3 deConvolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
#    result_merge = copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling)
    #conv1
    print('result_merge:',result_merge.shape)
    weights_1 = _weights("weights_1",
      shape=[3, 3, result_merge.get_shape()[3], k_down])
    biases_1 = _biases("biases_1", [k_down])
    result_conv_1 = tf.nn.conv2d(result_merge, weights_1,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
    result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",
      shape=[3, 3, result_relu_1.get_shape()[3], k_down])
    biases_2 = _biases("biases_2", [k_down])
    result_conv_2 = tf.nn.conv2d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
    result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), name='relu_2')
    result_dropout = tf.nn.dropout(x=result_relu_2, keep_prob=keep_prob)
    # unsample
    result = tf.cast(tf.image.resize_bilinear(tf.cast(result_dropout,tf.float32),[result_dropout.get_shape()[1]*2,result_dropout.get_shape()[2]*2]),tf.float32)
    
    weights_3 = _weights("weights_3",
      shape=[1, 1, result.get_shape()[3], k_up])
    biases_3 = _biases("biases_3", [k_up])
    result_conv_3 = tf.nn.conv2d(result, weights_3,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
    result_relu = tf.nn.relu(tf.nn.bias_add(result_conv_3, biases_3, name='add_bias'), name='relu_3')
    '''                        
    weights_3 = _weights("weights_3",
      shape=[2, 2, k_up, result_relu_2.get_shape()[3]])
    biases_3 = _biases("biases_3", [k_up])
    output_size = result_relu_2.get_shape()[1]*2
    output_shape = [result_relu_2.get_shape()[0], output_size, output_size, k_up]
    result_up_conv = tf.nn.conv2d_transpose(result_relu_2, weights_3,output_shape=output_shape,
                                      strides=[1, 2, 2, 1], padding='VALID',name = 'Up_Sample')
    result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up_conv, biases_3, name='add_bias'), name='relu_3')
    '''
		# dropout
    return result_relu
def layer_last(result_from_contract_layer,result_from_upsampling, k_down,num_class,reuse=False, keep_prob= 1, is_training=True, name='layer_last'):
  """ 2 3x3 Convolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    result_merge = copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling)
    #conv1
    weights_1 = _weights("weights_1",
      shape=[3, 3, result_merge.get_shape()[3], k_down])
    biases_1 = _biases("biases_1", [k_down])
    result_conv_1 = tf.nn.conv2d(result_merge, weights_1,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), is_training, 'batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",
      shape=[3, 3, result_relu_1.get_shape()[3], k_down])
    biases_2 = _biases("biases_2", [k_down])
    result_conv_2 = tf.nn.conv2d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_training, 'batch',name = 'conv2')
    
    #resnet cov
    weights_11 = _weights("weights_11",
      shape=[1, 1, result_merge.get_shape()[3], k_down])
    res_input = tf.nn.conv2d(result_merge, weights_11,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_11')                                                                                   
    resnormed = tf.add(normalized2,res_input)
    
    result_relu_2 = tf.nn.relu(resnormed, name='relu_2')
    
    #last conv
    weights_3 = _weights("weights_3",
      shape=[1, 1, k_down, num_class])
    biases_3 = _biases("biases_3", [num_class])
    result_conv_3 = tf.nn.conv2d(result_relu_2, weights_3,
				                              strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
    prediction = tf.nn.bias_add(result_conv_3, biases_3, name='add_bias')
    return prediction
def layer_last2(result_from_contract_layer,result_from_upsampling, k_down,num_class,reuse=False, keep_prob= 1, is_training=True, name='layer_last'):
  """ 2 3x3 Convolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    result_merge = copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling)
    
    w_1 = tf.reduce_mean(tf.reduce_mean(result_merge, axis=1, keep_dims=True),axis=2, keep_dims=True)
    weights_11 = _weights("weights_11",shape=[1, 1, result_merge.get_shape()[3],result_merge.get_shape()[3]],init = None)
    w_1 = tf.nn.conv2d(w_1, weights_11,strides=[1, 1, 1, 1], padding='SAME', name='conv_11')
    
    w_2 = tf.reduce_max(tf.reduce_max(result_merge, axis=1, keep_dims=True),axis=2, keep_dims=True)                                                                               
    weights_12 = _weights("weights_12",shape=[1, 1, result_merge.get_shape()[3],result_merge.get_shape()[3]],init = None)
    w_2 = tf.nn.conv2d(w_2, weights_12,strides=[1, 1, 1, 1], padding='SAME', name='conv_12')    
    w = w_1+w_2    
    w_norm = _norm(w, is_training, 'batch',name = 'w_norm')
    w = tf.nn.sigmoid(w_norm,name = 'sig')
    result = tf.multiply(result_merge, w) 
    
    w_3 = tf.reduce_mean(result,axis=3, keep_dims=True)
    w_4 = tf.reduce_max(result,axis=3, keep_dims=True)   
    w_34 = tf.concat(values=[w_3,w_4], axis=3)     
    weights_34 = _weights("weights_34",shape=[w_34.get_shape()[1], w_34.get_shape()[2], w_34.get_shape()[3],1],init = None)
    biases_34 = _biases("biases_34", [1],init = None)
    w2 = tf.nn.conv2d(w_34, weights_34,strides=[1, 1, 1, 1], padding='SAME', name='conv_34') 
    w34_norm = _norm(tf.nn.bias_add(w2, biases_34, name='add_bias'), is_training, 'batch',name = 'w34norm')  
    w2 = tf.nn.sigmoid(w34_norm,name = 'sig2')
    result2 = tf.multiply(result, w2)
    '''
    #conv1
#    input = tf.squeeze(input,1)
    weights_1 = _weights("weights_1",
      shape=[3, 3, result.get_shape()[3], k],init = None)
    biases_1 = _biases("biases_1", [k],init = None)
    result_conv_1 = tf.nn.conv2d(result, weights_1,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
    normalized1 = _norm(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), is_training, 'batch',name = 'conv1')
    result_relu_1 = tf.nn.relu(normalized1, name='relu_1')
    '''
    #conv2
    weights_2 = _weights("weights_2",
      shape=[1, 1, result2.get_shape()[3], k_down],init = None)
    biases_2 = _biases("biases_2", [k_down],init = None)
    result_conv_2 = tf.nn.conv2d(result2, weights_2,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
    normalized2 = _norm(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), is_training, 'batch',name = 'conv2')
    
    #resnet cov
    weights_11 = _weights("weights_res",
      shape=[1, 1, result.get_shape()[3], k_down])
    res_input = tf.nn.conv2d(result, weights_11,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_res')                                                                                   
    resnormed = tf.add(normalized2,res_input)
    
    result_relu_2 = tf.nn.relu(resnormed, name='relu_2')
#    result_dropout = tf.nn.dropout(x=result_relu_2, keep_prob=keep_prob)

    #last conv
    weights_4 = _weights("weights_4",
      shape=[1, 1, k_down, num_class])
    biases_4 = _biases("biases_4", [num_class])
    result_conv_4 = tf.nn.conv2d(result_relu_2, weights_4,
				                              strides=[1, 1, 1, 1], padding='VALID', name='conv_4')
    prediction = tf.nn.bias_add(result_conv_4, biases_4, name='add_bias')
    return prediction
def layer_lastGAN(result_merge, k_down,num_class,reuse=False, keep_prob= 1, is_training=True, name='layer_up'):
  """ 2 3x3 Convolution-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    keep_prob: dropout rate 
    name: string, e.g. 'layer_1'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
#    result_merge = copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling)
    #conv1
    weights_1 = _weights("weights_1",
      shape=[3, 3, result_merge.get_shape()[3], k_down])
    biases_1 = _biases("biases_1", [k_down])
    result_conv_1 = tf.nn.conv2d(result_merge, weights_1,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
    result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, biases_1, name='add_bias'), name='relu_1')
    
    #conv2
    weights_2 = _weights("weights_2",
      shape=[3, 3, result_relu_1.get_shape()[3], k_down])
    biases_2 = _biases("biases_2", [k_down])
    result_conv_2 = tf.nn.conv2d(result_relu_1, weights_2,
				                              strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
    result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, biases_2, name='add_bias'), name='relu_2')
    
    #last conv
    weights_3 = _weights("weights_3",
      shape=[1, 1, k_down, num_class])
    biases_3 = _biases("biases_3", [num_class])
    result_conv_3 = tf.nn.conv2d(result_relu_2, weights_3,
				                              strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
    prediction = tf.nn.bias_add(result_conv_3, biases_3, name='add_bias')
    return prediction


def copy_and_crop_and_merge(result_from_contract_layer,result_from_upsampling):
  return tf.concat(values=[result_from_contract_layer, result_from_upsampling], axis=-1)
  
def _weights(name, shape, mean=0.0, stddev=0.02,init = None):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  '''
  if init is None:
    var = tf.get_variable(
      name, shape,
      initializer=tf.random_normal_initializer(
        mean=mean, stddev=stddev, dtype=tf.float32))
  else:
    if len(shape) == 5:
      real_init = tf.constant_initializer(np.repeat(init[np.newaxis,:],shape[0],axis=0))
      var = tf.get_variable(name, shape,initializer=real_init)
    else:
      if shape[3] == 1024:
        real_init = tf.constant_initializer(np.repeat(init,2,axis=3))
        var = tf.get_variable(name, shape,initializer=real_init)
      else:
        var = tf.get_variable(name, shape,initializer=tf.constant_initializer(init))
  '''
  var = tf.get_variable(
      name, shape,dtype=tf.float32,
      initializer=tf.random_normal_initializer(
        mean=mean, stddev=stddev, dtype=tf.float32)) 
  return var


def _biases(name, shape, constant=0.0,init = None):
  """ Helper to create an initialized Bias with constant
  """
  '''
  if init is None:
    var = tf.get_variable(name, shape,initializer=tf.constant_initializer(constant))
  else:
    if len(shape) == 5:
      var = tf.get_variable(name, shape,initializer=tf.constant_initializer(init))
    else:
      if shape[0] == 1024:
        real_init = tf.constant_initializer(np.repeat(init,2,axis=0))
        var = tf.get_variable(name, shape,initializer=real_init)
      else:
        var = tf.get_variable(name, shape,initializer=tf.constant_initializer(init))
  '''
  var = tf.get_variable(name, shape,dtype=tf.float32,initializer=tf.constant_initializer(constant))
  return var
#  return tf.get_variable(name, shape,
#            initializer=tf.constant_initializer(constant))

def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)

def _norm(input, is_training, norm='instance',name = 'conv'):
  """ Use Instance Normalization or Batch Normalization or None
  """
  if norm == 'instance':
    return _instance_norm(input)
  elif norm == 'batch':
    return _batch_norm(input, is_training,bname = name)
  else:
    return input

def _batch_norm(input, is_training,bname = 'conv'):
  """ Batch Normalization
  """
  with tf.variable_scope(bname+"_batch_norm"):
    '''
    return tf.contrib.layers.batch_norm(input,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=is_training)
    '''
    return tf.layers.batch_normalization(input,
             #                           decay=0.9,
                                        scale=True,
             #                           updates_collections=None,
                                        training=is_training)

def _instance_norm(input):
  """ Instance Normalization
  """
  with tf.variable_scope("instance_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)
