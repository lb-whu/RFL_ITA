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

from metric import dice_score2 
from metric import sensitivity 
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
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('DataPath2', '/home/libo/train_br20_norm/whole_orig4/',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('valPath', '/home/libo/train_br20_norm/val/',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('GraphPath', '/home/libo/regbrain/pretrained/reg.pb',
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('model_path', '/home/libo/segment/largefov.npy',
                       'vgg16 path, default:')
tf.flags.DEFINE_string('valPath2', './val/',
                       'validata path, default:')
#tf.flags.DEFINE_string('Y', 'facade/tfrecords/Y.tfrecords',
                       #'Y tfrecords file for training, default:')
tf.flags.DEFINE_integer('NUM_ID',25885,
                       'X tfrecords file for training, default:')
tf.flags.DEFINE_string('load_model','checkpoints/20211209-0911',
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
    # Shuffle the ordering of all image files in order to guarantee
    # random ordering of the images with respect to label in the
    # saved TFRecord files. Make the randomization repeatable.
    shuffled_index = list(range(len(file_paths)))
#    random.seed(12345)
    random.shuffle(shuffled_index)

    file_paths = [file_paths[i] for i in shuffled_index]

  return file_paths
def export_graph(model_name1,model_name2):
  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
#    checkpoints_dir2 = "checkpoints2/" + FLAGS.load_model.lstrip("checkpoints2/")
  else:
    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    checkpoints_dir = "checkpoints/{}".format(current_time)
#    checkpoints_dir2 = "checkpoints2/{}".format(current_time)
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
        model_path = FLAGS.model_path,
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
    Unet_Model.model()
    
#    if XtoY:
    src = tf.placeholder(tf.float32,
        shape=[1,FLAGS.D, FLAGS.H, FLAGS.W, 4],name='src')
    tgt = tf.placeholder(tf.float32,
        shape=[1,FLAGS.D, FLAGS.H, FLAGS.W, 4],name='tgt')
    label = tf.placeholder(tf.float32,
        shape=[1,FLAGS.D, FLAGS.H, FLAGS.W, 5],name='label')
    label_t = tf.placeholder(tf.float32,
        shape=[1,FLAGS.D, FLAGS.H, FLAGS.W, 5],name='label_t')    
    y, flow,l = Unet_Model.Unet1.sample(src,tgt,label,label_t)
    print('y:',y.shape)
    print('flow:',flow.shape)
    print('l:',l.shape)
    
    y = tf.identity(y, name='y')
    flow = tf.identity(flow, name='flow')
    l = tf.identity(l, name='l')

#    print(out_image)
    
#    restore_saver = tf.train.Saver()
#    export_saver = tf.train.Saver()
#    restore_saver = tf.train.import_meta_graph("/home/libo/segment/segTumor/checkpoints/20211209-0911/model.ckpt-3003.meta")
  with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    latest_ckpt = tf.train.latest_checkpoint(FLAGS.load_model)
    restore_saver.restore(sess, latest_ckpt)

    output_graph_y = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [y.op.name])
    tf.train.write_graph(output_graph_y, 'pretrained', model_name1, as_text=False)
    
    output_graph_l = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [l.op.name])
    tf.train.write_graph(output_graph_l, 'pretrained', model_name2, as_text=False)
    '''
    f_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [c1.op.name,c2.op.name,c3.op.name,c4.op.name,feature.op.name])
    tf.train.write_graph(f_graph_def, 'pretrained', 'f_bs1.pb', as_text=False)
    
    f_down_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [f_down.op.name])
    tf.train.write_graph(f_down_graph_def, 'pretrained', 'f_down.pb', as_text=False)
    '''

def main(unused_argv):
  print('Export XtoY model...')
  export_graph('trans_3001.pb','trans_3002.pb')
#  print('Export YtoX model...')
#  export_graph(FLAGS.YtoX_model, XtoY=False)

if __name__ == '__main__':
  tf.app.run()
