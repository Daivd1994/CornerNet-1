import tensorflow as tf
import os
from utils.get_data import Image_data
from network import NetWork
from config import cfg
class Train():
    def __init__(self):
        self.net=NetWork()
        self.data=Image_data('trainval')
        self.data.inupt_producer()
        self.gpus=[0,1]
        self.batch_i=2
        self.batch_size=self.batch_i*len(self.gpus)
        self.save_pre_every=int(self.data.num_image/self.batch_size)+1
        self.num_steps=int(self.save_pre_every*100+1)
        self.lr=cfg.learning_rate
    def train(self,sess):
        coord = tf.train.Coordinator()
        images, tags_tl, tags_br,heatmaps_tl, heatmaps_br, tags_mask, offsets_tl, offsets_br,boxes=self.data.get_batch_data(self.batch_size)
        tower_grads = []
        optim=tf.train.AdamOptimizer(self.lr)
        is_training=tf.constant(True)
        for i in range(len(self.gpus)):
            with tf.device('/gpu:%d'%i):
                with tf.variable_scope('Tower_%d'%i) as scope:
                    next_imgs=images[i*self.batch_i:(i+1)*self.batch_i]
                    next_tags_tl=tags_tl[i*self.batch_i:(i+1)*self.batch_i]
                    next_tags_br=tags_br[i*self.batch_i:(i+1)*self.batch_i]
                    next_heatmaps_tl=heatmaps_tl[i*self.batch_i:(i+1)*self.batch_i]
                    next_heatmaps_br=heatmaps_br[i*self.batch_i:(i+1)*self.batch_i]
                    next_tags_mask=tags_mask[i*self.batch_i:(i+1)*self.batch_i]
                    next_offsets_tl=offsets_tl[i*self.batch_i:(i+1)*self.batch_i]
                    next_offsets_br=offsets_br[i*self.batch_i:(i+1)*self.batch_i]
                    outs=self.net.corner_net(next_imgs,next_tags_tl,next_tags_br,is_training=is_training)
                    loss=self.net.loss(outs,[next_heatmaps_tl,next_heatmaps_br,next_tags_mask,next_offsets_tl,next_offsets_br])
                    trainable_variable = tf.trainable_variables()
                    grads = optim.compute_gradients(loss, var_list=trainable_variable)
                    tower_grads.append(grads)
        grads_ave = average_gradients(tower_grads)
        update=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update):
            train_op = optim.apply_gradients(grads_ave)
        saver = tf.train.Saver(max_to_keep=100)
        loader = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess.run(init)
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        for step in range(self.num_steps):
            start_time = time.time()
            _,loss_=sess.run(train_op,loss)
            if step % self.save_pre_every == 0:
                saver.save(sess, SNAPSHOT_DIR, step)
            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f},({:.3f} sec/step)'.format(step,loss_,duration))
