# -*- coding: utf-8 -*-
#from skimage import transform as trans
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from scipy.ndimage.interpolation import zoom
def getMeanVarIMG(ds,ls,dt,lt,l_num):
    ds2 = ds[:,:,:,np.newaxis]  
    snum_img = np.sum(ls,axis=(0,1,2))
    aver_cls = np.sum(ds2 *ls,axis = (0,1,2))/snum_img
    m_img= np.sum(aver_cls*ls,axis = (3))
    var = (ds-m_img)**2
    var = var[:,:,:,np.newaxis] 
    var_img = np.sqrt(np.sum(np.sum(var*ls,axis = (0,1,2))*ls/snum_img,axis = (3)))
    
    dt2 = dt[:,:,:,np.newaxis]  
    tnum_img = np.sum(lt,axis=(0,1,2))
    tm_img= (np.sum(dt2 *lt,axis = (0,1,2)))/tnum_img
    ttm_img = np.sum(tm_img*lt,axis = (3))
    tsm_img = np.sum(tm_img*ls,axis = (3))
    tvar = (dt-ttm_img)**2
    tvar = tvar[:,:,:,np.newaxis] 
    tvar_img = np.sqrt(np.sum(np.sum(tvar*lt,axis = (0,1,2))*ls/tnum_img,axis = (3)))
    t_img = tvar_img*(ds-m_img)/(var_img+1e-8) + tsm_img
    return t_img
    
def getMeanVarIMG2(ds,ls,dt,lt,l_num):

    snum_img = tf.sum(ls,axis=(0,1,2,3))
    aver_cls = tf.sum(ds *ls,axis = (0,1,2,3))/snum_img
    m_img= tf.sum(aver_cls*ls,axis = (4))
    var = (ds-m_img)**2
    var = tf.expand_dims(var,4) 
    var_img = tf.sqrt(tf.sum(tf.sum(var*ls,axis = (0,1,2,3))*ls/snum_img,axis = (4)))
     
    tnum_img = tf.sum(lt,axis=(0,1,2,3))
    tm_img= (tf.sum(dt *lt,axis = (0,1,2,3)))/tnum_img
    ttm_img = tf.sum(tm_img*lt,axis = (4))
    tsm_img = tf.sum(tm_img*ls,axis = (4))
    tvar = (dt-ttm_img)**2
    tvar = tf.expand_dims(tvar,4) 
    tvar_img = tf.sqrt(tf.sum(tf.sum(tvar*lt,axis = (0,1,2,3))*ls/tnum_img,axis = (4)))
    t_img = tvar_img*(ds-m_img)/(var_img+1e-8) + tsm_img
    return t_img
def zm(x,y):
    D,W,H = x.shape
    datax2 = np.zeros((D,W,H),dtype = np.float32)
    datay2 = np.zeros((D,W,H),dtype = np.float32)
    datax = zoom(x, zoom = [0.9,0.9,1], order=0)
    datay = zoom(y, zoom = [0.9,0.9,1], order=0)
    datax2[int(D*0.05):int(D*0.95),int(W*0.05):int(W*0.95),:] = datax
    datay2[int(D*0.05):int(D*0.95),int(W*0.05):int(W*0.95),:] = datay
    return datax2,datay2    
def crop(x):
    D,W,H,f = x.shape
#    print('W',x.shape)
    for i in range(0,D):
        if np.max(x[i,:,:,:]) > 0:
            ind_ld = i
            break
    for i in range(D-1,-1,-1):
        if np.max(x[i,:,:,:]) > 0:
            ind_rd = i + 1
            break
    for i in range(0,W):
        if np.max(x[:,i,:,:]) > 0:
            ind_lw = i
            break
    for i in range(W-1,-1,-1):
        if np.max(x[:,i,:,:]) > 0:
            ind_rw = i + 1
            break
    for i in range(0,H):
        if np.max(x[:,:,i,:]) > 0:
            ind_lh = i
            break
    for i in range(H-1,-1,-1):
        if np.max(x[:,:,i,:]) > 0:
            ind_rh = i + 1
            break
    return [ind_ld,ind_rd,ind_lw,ind_rw,ind_lh,ind_rh],ind_rd - ind_ld,ind_rw - ind_lw,ind_rh -ind_lh 

def cropTmr(x,y,xt,yt):
    D,W,H,f = x.shape
#    print('W',x.shape)
    for i in range(0,D):
        if (np.max(xt[i,:,:,:]) > 0)|(np.max(yt[i,:,:,:]) > 0):
            ind_ld = i
            break
    for i in range(D-1,-1,-1):
        if (np.max(xt[i,:,:,:]) > 0)|(np.max(yt[i,:,:,:]) > 0):
            ind_rd = i + 1
            break
    for i in range(0,W):
        if (np.max(xt[:,i,:,:]) > 0)|(np.max(yt[:,i,:,:]) > 0):
            ind_lw = i
            break
    for i in range(W-1,-1,-1):
        if (np.max(xt[:,i,:,:]) > 0)|(np.max(yt[:,i,:,:]) > 0):
            ind_rw = i + 1
            break
    for i in range(0,H):
        if (np.max(xt[:,:,i,:]) > 0)|(np.max(yt[:,:,i,:]) > 0):
            ind_lh = i
            break
    for i in range(H-1,-1,-1):
        if (np.max(xt[:,:,i,:]) > 0)|(np.max(yt[:,:,i,:]) > 0):
            ind_rh = i + 1
            break
    x[0:ind_ld,:,:,:]=0
    y[0:ind_ld,:,:,:]=0
    x[:,:,0:ind_lh,:]=0
    y[:,:,0:ind_lh,:]=0
    x[:,0:ind_lw,:,:]=0
    y[:,0:ind_lw,:,:]=0
    
    x[ind_rd+1:D+1,:,:,:]=0
    y[ind_rd+1:D+1,:,:,:]=0
    x[:,:,ind_rh+1:H+1,:]=0
    y[:,:,ind_rh+1:H+1,:]=0
    x[:,ind_rw+1:W+1,:,:]=0
    y[:,ind_rw+1:W+1,:,:]=0

#    print(ind_ld,ind_rd,ind_lw,ind_rw,ind_lh,ind_rh)
    return x.copy(),y.copy()
def L2HOT(y,label_num):
    label = np.zeros((y.shape[0],y.shape[1],y.shape[2],label_num),dtype = np.float32)
    for k in range(0,label_num):
        gt = y.copy()
        gt[gt == k] = 10
        gt[gt != 10] = 0
        gt[gt == 10] = 1
        label[:,:,:,k] =  gt
    return label
def cropback(x,box,origsize):
    img = np.zeros((origsize[0],origsize[1],origsize[2]),dtype = np.float32)
    img[box[0,0]:box[0,1],box[0,2]:box[0,3],box[0,4]:box[0,5]] = x
    return img

def aug(x,y,label_num):
    """
  aug images from input_dir
  Args:
    flag: 0--current 
  Returns:
    auged x
    """
    flag = np.random.randint(low = 0,high = 3, size=(1))
    rot = np.random.randint(low = -5,high = 5, size=(1))
          
    if flag == 0:
        img2 = np.reshape(x.copy(),(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
        l2 = np.reshape(y.copy(),(y.shape[0],y.shape[1],y.shape[2]*y.shape[3]))
        datax = trans.rotate(img2, rot)
        datay = trans.rotate(l2, rot)
        datax = np.reshape(datax,(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        datay = np.reshape(datay,(y.shape[0],y.shape[1],y.shape[2],y.shape[3]))
        if np.random.randint(low = 0,high = 2, size=(1)) == 1:
            datax = datax.transpose((1,2,0,3))
            datay = datay.transpose((1,2,0,3))
            datax2 = np.reshape(datax.copy(),(datax.shape[0],datax.shape[1],datax.shape[2]*datax.shape[3]))
            datay2 = np.reshape(datay.copy(),(datay.shape[0],datay.shape[1],datay.shape[2]*datay.shape[3]))        
            datax2 = np.fliplr(datax2)
            datay2 = np.fliplr(datay2)
            datax2 = np.reshape(datax2,(datax.shape[0],datax.shape[1],datax.shape[2],datax.shape[3]))
            datay2 = np.reshape(datay2,(datay.shape[0],datay.shape[1],datay.shape[2],datay.shape[3])) 
            datax = datax2.transpose((2,0,1,3))
            datay = datay2.transpose((2,0,1,3))
    else:
        if flag == 1:
            x = x.transpose((1,2,0,3)).copy()
            y = y.transpose((1,2,0,3)).copy()
        elif flag == 2:
            x = x.transpose((2,0,1,3)).copy()
            y = y.transpose((2,0,1,3)).copy()
        img2 = np.reshape(x.copy(),(x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
        l2 = np.reshape(y.copy(),(y.shape[0],y.shape[1],y.shape[2]*y.shape[3]))
        datax = trans.rotate(img2, rot)
        datay = trans.rotate(l2, rot)

        datax = np.reshape(datax,(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        datay = np.reshape(datay,(y.shape[0],y.shape[1],y.shape[2],y.shape[3]))
        if flag == 1:
            if np.random.randint(low = 0,high = 2, size=(1)) == 1:
                datax = np.fliplr(datax)
                datay = np.fliplr(datay)
            datax = datax.transpose((2,0,1,3))
            datay = datay.transpose((2,0,1,3))
        elif flag == 2:
            datax = datax.transpose((1,2,0,3))
            datay = datay.transpose((1,2,0,3))
    
    return datax,datay