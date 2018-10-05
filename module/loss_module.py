import numpy as np
import tensorflow as tf
def focal_loss(preds,gt):
    zeros=tf.zeros_like(gt)
    ones=tf.ones_like(gt)
    num_pos=tf.reduce_sum(tf.where(gt==1,ones,zeros))
    loss=0
    for pre in preds:
        pos_weight=tf.where(gt==1,ones-pre,zeros)
        neg_weight=tf.where(gt<1,pre,zeros)
        pos_loss=tf.reduce_sum(tf.log(pre) * (pos_weight ** 2))
        neg_loss=tf.reduce_sum(((1-gt)**4)*(neg_weight**2)*tf.log((1-pre)))
        loss=loss-(pos_loss+neg_loss)/num_obj
    return loss
def tag_loss(tag0, tag1, mask):
    #pull
    zeros=tf.zeros_like(mask)
    num  = tf.reduce_sum(mask)
    tag_mean = (tag0 + tag1) / 2
    tag0 = ((tag0 - tag_mean) ** 2) / (num + 1e-4)
    tag0_mask=tf.where(mask==1,tag0,zeros)
    tag0 = tf.reduce_sum(tag0_mask)
    tag1 = ((tag1 - tag_mean)** 2) / (num + 1e-4)
    tag1_mask=tf.where(mask==1,tag1,zeros)
    tag1 = tf.reduce_sum(tag1_mask)
    pull = tag0 + tag1
    #push
    mask=tf.reshape(mask,(tf.shape(mask)[0],1,tf.shape(mask)[1]))+tf.reshape(mask,(tf.shape(mask)[0],tf.shape(mask)[1],1))
    zeros=tf.zeros_like(mask)
    ones=tf.ones_like(mask)
    mask=tf.where(mask==2,ones,zeros)
    num2=num*(num-1)
    dist=tf.reshape(tag_mean,(tf.shape(tag_mean)[0],1,tf.shape(tag_mean)[1]))-tf.reshape(tag_mean,(tf.shape(tag_mean)[0],tf.shape(tag_mean)[1],1))
    dist=1-tf.abs(dist)
    dist=tf.nn.relu(dist)
    dist=dist-1 / (num + 1e-4)
    dist=dist / (num2 + 1e-4)
    dist=mask*dist
    push=tf.reduce_sum(dist)
    return pull, push

def offset_loss(regr, gt_regr, mask):
    num  = tf.reduce_sum(mask)
    mask = tf.stack((mask,mask),-1)
    regr_loss = smooth_l1_loss(regr, gt_regr)
    regr_loss = regr_loss / (num + 1e-4)
    regr_loss=regr_loss*mask
    return regr_loss
def smooth_l1_loss(pred,targets):
    diff = pred -targets
    abs_diff = tf.abs(diff)
    smoothL1_sign =tf.to_float(tf.less(abs_diff, 1))
    loss = tf.pow(diff, 2) * 0.5 * smoothL1_sign + (abs_diff - 0.5) * (1. - smoothL1_sign)
    return loss


