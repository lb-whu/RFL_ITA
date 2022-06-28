import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow as tf
from modelac import UnetModel
from datetime import datetime
import os
import logging
import aug
import random
#import string
import numpy as np
import scipy.io
import time
import SimpleITK as sitk

try:
  from os import scandir
except ImportError:
  # Python 2 polyfill module
  from scandir import scandir
import scipy.io as sio   
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
FLAGS = tf.flags.FLAGS
class_weight = [0.5,1,1,1]
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('D', 136, 'image size, default: 240')
tf.flags.DEFINE_integer('H', 192, 'image patch_size, default: 120')
tf.flags.DEFINE_integer('W', 152, 'image patch_size, default: 120')
tf.flags.DEFINE_bool('use_lsgan', True,
                     'use lsgan (mean squared error) or cross entropy loss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
                       '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_float('learning_rate', 1e-4,
                      'initial learning rate for Adam, default: 0.0002')
tf.flags.DEFINE_float('beta1', 0.5,
                      'momentum term of Adam, default: 0.5')
tf.flags.DEFINE_float('pool_size', 50,
                      'size of image buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 32,
                        'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_integer('num_class', 4,
                        'number of class, default: 9')
tf.flags.DEFINE_string('DataPath', '/home/libo/train_br20_norm/whole_orig3/',
                       'hard samples:')
tf.flags.DEFINE_string('DataPath2', '/home/libo/train_br20_norm/whole_orig4/',
                       'easy samples:')
tf.flags.DEFINE_string('valPath', '/home/libo/train_br20_norm/val/',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('GraphPath', '/home/libo/regbrain/pretrained/reg.pb',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('valPath2', './val/',
                       'validata path, default:')
tf.flags.DEFINE_integer('NUM_ID',25885,
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('load_model',None,
                        'folder of saved model that you wish to continue training (e.g. 20170602-1936), default: None')
                        
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
OutputPath = '/home/libo/segment/segTumor/val/'
def data_reader(input_dir1,input_dir2, shuffle=True):
  """Read images from input_dir then shuffle them
  Args:
    input_dir: string, path of input dir, e.g., /path/to/dir
  Returns:
    file_paths: list of strings
  """
  file_paths = []

  for img_file in scandir(input_dir1):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.png') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.mat') and img_file.is_file():
      file_paths.append(img_file.path)
  for img_file in scandir(input_dir2):
    if img_file.name.endswith('.jpg') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.png') and img_file.is_file():
      file_paths.append(img_file.path)
    if img_file.name.endswith('.mat') and img_file.is_file():
      file_paths.append(img_file.path)

  if shuffle:

    shuffled_index = list(range(len(file_paths)))
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths
def train():
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
    try:
      os.makedirs(checkpoints_dir)
    except os.error:
      pass
  '''
  grp2 = tf.Graph()
  with grp2.as_default():
    reader = Reader2(FLAGS.DataPath, name='U',
        image_size=FLAGS.image_size, batch_size=FLAGS.batch_size)
    name,x,y= reader.feed()
  '''
  print('~~~~~~~~~~~~~~~~~~~~~!!!!')
  graph = tf.Graph()
  with graph.as_default():
    Unet_Model = UnetModel(
        DataPath=FLAGS.DataPath,
        GraphPath = FLAGS.GraphPath,
        batch_size=FLAGS.batch_size,
        D=FLAGS.D,
        H=FLAGS.H,
        W=FLAGS.W,
        use_lsgan=FLAGS.use_lsgan,
        norm=FLAGS.norm,
        learning_rate=FLAGS.learning_rate,
        beta1=FLAGS.beta1,
        ngf=FLAGS.ngf,
        num_class = FLAGS.num_class
    )
    loss_cc,s_t,l_st= Unet_Model.model()
    optimizers = Unet_Model.optimize(loss_cc)
    
    saver = tf.train.Saver()
  tf_config = tf.ConfigProto() 
  with tf.Session(graph=graph,config = tf_config) as sess:
    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      print(meta_graph_path)
      step = int(meta_graph_path.split("-")[2].split(".")[0])
    else:
      sess.run(tf.global_variables_initializer())
      step = 0
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
      loss_2000 = 0
      loss_1 = 0
      loss_2 = 0
      loss_echo = 0
      echo = 369*2
      cst = 123*2
      count = 0
      list = os.listdir(FLAGS.DataPath) 
      while not coord.should_stop():
  #    while True:
        
#        print('~!!!!!!!!!!!!!')
#        a = time.time()
#        x_val,y_val = sess.run([x,y])
        
        file_paths_X = data_reader(FLAGS.DataPath,FLAGS.DataPath2)
        file_paths_X2 = data_reader(FLAGS.DataPath,FLAGS.DataPath2)
#        val_files = data_reader(FLAGS.valPath2)
        num_hard = len(file_paths_X)
        i_hard = 0
        while(1):
          if not coord.should_stop():
            datas = file_paths_X[i_hard]
            datat = file_paths_X2[i_hard]
            if datas == datat:
              if (i_hard + 1) == num_hard:
                i_hard = 0
                file_paths_X = data_reader(FLAGS.DataPath,FLAGS.DataPath2)
                file_paths_X2 = data_reader(FLAGS.DataPath,FLAGS.DataPath2)
                step += 1
              else:
                step += 1
                i_hard += 1
              continue
            datass=sio.loadmat(datas) 
            datatt=sio.loadmat(datat) 
            d_s = np.float32(datass['img'])
            d_s = np.squeeze(d_s)
            
            l_s = np.float32(datass['label'])
            l_s = np.squeeze(l_s)
            

            d_t = np.float32(datatt['img'])
            d_t = d_t.astype(np.float32)            
            l_t = np.float32(datatt['label'])
            l_t = np.squeeze(l_t)
            lt_size = np.size(np.unique(l_t[:]))
#            print(lt_size)
            if lt_size < 5:
              if (i_hard + 1) == num_hard:
                i_hard = 0
                file_paths_X = data_reader(FLAGS.DataPath,FLAGS.DataPath2)
                file_paths_X2 = data_reader(FLAGS.DataPath,FLAGS.DataPath2)
                step += 1
              else:
                step += 1
                i_hard += 1
              continue
            l_s_h = aug.L2HOT(l_s,5)
            l_t_h = aug.L2HOT(l_t,5)
            
            l_s = l_s[np.newaxis,:,:,:,np.newaxis]
            d_s = d_s[np.newaxis,:,:,:,:]
            d_t = d_t[np.newaxis,:,:,:,:]

            l_s_h = l_s_h[np.newaxis,:,:,:,:]
            l_t_h = l_t_h[np.newaxis,:,:,:,:]

            _,loss_cc_val,pred_val,l_st_val= (sess.run([optimizers,loss_cc,s_t,l_st], feed_dict={Unet_Model.ds: d_s,Unet_Model.dt: d_t,Unet_Model.ls: l_s_h,Unet_Model.lt: l_t_h}))
            loss_2000 = loss_2000 + loss_cc_val
            loss_1 = loss_1 + loss_cc_val
            loss_2 = loss_2 + loss_cc_val
            loss_echo = loss_echo + loss_cc_val
            i_hard = i_hard + 1
            
            if i_hard == num_hard:
              i_hard = 0
              file_paths_X = data_reader(FLAGS.DataPath,FLAGS.DataPath2)
              file_paths_X2 = data_reader(FLAGS.DataPath,FLAGS.DataPath2)
            step += 1
                     
          else:
            break
          if count % cst == 0:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logger.info('-----------Step %d:-------------' % step)
  #          logger.info('  loss_225  : {}'.format(loss_2000))
            logger.info('  loss_1  : {}'.format(loss_1))
            logger.info('  loss_2  : {}'.format(loss_2))
  #          loss_2000 = 0
            loss_1 = 0
            loss_2 = 0

          if count % echo == 0:
            save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
            logger.info('-----------Step %d:-------------' % step)
            logger.info('!!!!!!!!!!!!loss_2000!!!!!!!!!!!!!!!!!   : {}'.format(loss_2000)) 
            logger.info('!!!!!!!!!!!!LOSS_ECHO!!!!!!!!!!!!!!!!!   : {}'.format(loss_echo)) 
  #          logger.info('!!!!!!!!!!!!LOSS_ECHO!!!!!!!!!!!!!!!!!   : {}'.format(loss_echo)) 
            loss_2000 = 0
            loss_echo = 0                   
          count += 1            

 

  
           
    except KeyboardInterrupt:
      logging.info('Interrupted')
      coord.request_stop()
    except Exception as e:
      coord.request_stop(e)
    finally:
      save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)
      logging.info("Model saved in file: %s" % save_path)
      # When done, ask the threads to stop.
      coord.request_stop()
      coord.join(threads)

def main(unused_argv):
  train()

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  tf.app.run()
