import tensorflow as tf
from model import Model
from module.loss_module import focal_loss,tag_loss,offset_loss
from module.forward_module import nms,top_k,map_to_vector,expand_copy
class NetWork():
    def __init__(self,pull_weight=0.1, push_weight=0.1, offset_weight=1):
        self.n_deep  = 5
        self.n_dims  = [256, 256, 384, 384, 384, 512]
        self.n_res   = [2, 2, 2, 2, 2, 4]
        self.out_dim = 80
        self.model=Model()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.offset_weight = offset_weight
        self.focal_loss  = focal_loss
        self.tag_loss     = tag_loss
        self.offset_loss   = offset_loss
    def corner_net(self,img,gt_tag_tl=None,gt_tag_br=None,is_training=True,scope='CornerNet')
        with tf.variable_scope(scope):
            outs=[]
            start_layer=self.model.start_conv(img,is_training=is_training)#[b,128,128,256]
            with tf.variable_scope('inter_supervise'):
                hourglass_1=self.model.hourglass(start_layer,self.n_deep,self.n_res,self.n_dims,is_training=is_training)#[b,128,128,256]
                hinge_is=self.model.hinge(hourglass_1,256,256,is_training=is_training)
                top_left_is,bottom_right_is=self.model.corner_pooling(hinge_is,256,256,is_training=is_training)
                #top_left
                heat_tl_is=self.model.heat(top_left_is,256,self.out_dim,scope='heat_tl')
                tag_tl_is=self.model.tag(top_left_is,256,1,scope='tag_tl')
                if not gt_tag_tl is None:
                    tag_tl_is=map_to_vector(tag_tl_is,gt_tag_tl)
                offset_tl_is=self.model.offset(top_left_is,256,2,scope='offset_tl')
                if not gt_tag_tl is None:
                    offset_tl_is=map_to_vector(offset_tl_is,gt_tag_tl)
                #bottom_right
                heat_br_is=self.model.heat(bottom_right_is,256,self.out_dim,scope='heat_br')
                tag_br_is=self.model.tag(bottom_right_is,256,1,scope='tag_br')
                if not gt_tag_br is None:
                    tag_br_is=map_to_vector(tag_br_is,gt_tag_br)
                offset_br_is=self.model.offset(bottom_right_is,256,2,scope='offset_br')
                if not gt_tag_br is None:
                    offset_br_is=map_to_vector(offset_br_is,gt_tag_br)


            with tf.variable_scope('master_branch'):
                inter=self.model.inter(start_layer,hinge_is,256,is_training=is_training)
                hourglass_2=self.model.hourglass(inter,self.n_deep,self.n_res,self.n_dims,is_training=is_training)#[b,128,128,256]
                hinge=self.model.hinge(hourglass_2,256,256,is_training=is_training)
                top_left,bottom_right=self.model.corner_pooling(hinge,256,256,is_training=is_training)
                #top_left
                heat_tl=self.model.heat(top_left,256,self.out_dim,scope='heat_tl')
                tag_tl=self.model.tag(top_left,256,1,scope='tag_tl')
                if not gt_tag_tl is None:
                    tag_tl=map_to_vector(tag_tl,gt_tag_tl)
                offset_tl=self.model.offset(top_left,256,2,scope='offset_tl')
                if not gt_tag_tl is None:
                    offset_tl=map_to_vector(offset_tl,gt_tag_tl)
                #bottom_right
                heat_br=self.model.heat(bottom_right,256,self.out_dim,scope='heat_br')
                tag_br=self.model.tag(bottom_right,256,1,scope='tag_br')
                if not gt_tag_br is None:
                    tag_br=map_to_vector(tag_br,gt_tag_br)
                offset_br=self.model.offset(bottom_right,256,2,scope='offset_br')
                if not gt_tag_br is None:
                    offset_br=map_to_vector(offset_br,gt_tag_br)

            outs=[heat_tl_is,heat_br_is,tag_tl_is,tag_br_is,offset_tl_is,offset_br_is,heat_tl,heat_br,tag_tl,tag_br,offset_tl,offset_br]
            return outs
    def loss(self,outs,targets,scope='loss'):
        with tf.variable_scope(scope):
            stride = 6
            heats_tl = outs[0::stride]
            heats_br = outs[1::stride]
            tags_tl  = outs[2::stride]
            tags_br  = outs[3::stride]
            offsets_tl = outs[4::stride]
            offsets_br = outs[5::stride]

            gt_heat_tl = targets[0]
            gt_heat_br = targets[1]
            gt_mask    = targets[2]
            gt_offset_tl = targets[3]
            gt_offset_br = targets[4]

            # focal loss
            focal_loss = 0

            heats_tl = [tf.clip_by_value(tf.nn.sigmoid(tl),min=1e-4, max=1-1e-4) for tl in heats_tl]
            heats_br = [tf.clip_by_value(tf.nn.sigmoid(br),min=1e-4, max=1-1e-4) for tl in heats_br]

            focal_loss += self.focal_loss(heats_tl, gt_heat_tl)
            focal_loss += self.focal_loss(heats_br, gt_heat_br)

            # tag loss
            pull_loss = 0
            push_loss = 0

            for tag_tl, tag_br in zip(tags_tl, tags_br):
                pull, push = self.tag_loss(tag_tl, tag_br, gt_mask)
                pull_loss += pull
                push_loss += push
            pull_loss = self.pull_weight * pull_loss
            push_loss = self.push_weight * push_loss

            offset_loss = 0
            for offset_tl, offset_br in zip(offsets_tl, offsets_br):
                offset_loss += self.offset_loss(offset_tl, gt_offset_tl, gt_mask)
                offset_loss += self.offset_loss(offset_br, gt_offset_br, gt_mask)
            offset_loss = self.offset_weight * offset_loss

            loss = (focal_loss + pull_loss + push_loss + offset_loss) / len(heats_tl)
            return loss
    def decode(self,heat_tl,heat_br,tag_tl,tag_br,offset_tl,offset_br,k=100,ae_threshold=0.5,num_dets=1000):
        batch=tf.shape(heat_br)[0]
        heat_tl=tf.nn.sigmoid(heat_tl)
        heat_br=tf.nn.sigmoid(heat_br)
        #nms
        heat_tl=nms(heat_tl)
        heat_br=nms(heat_br)
        value_tl,position_tl,class_tl,y_tl,x_tl=top_k(heat_tl,k)
        value_br,position_br,class_br,y_br,x_br=top_k(heat_br,k)

        #expand to square
        offset_tl=map_to_vector(offset_tl,position_tl)
        offset_br=map_to_vector(offset_br,position_br)

        x_tl=x_tl+offset_tl[:,:,0]
        y_tl=y_tl+offset_tl[:,:,1]
        x_br=x_br+offset_br[:,:,0]
        y_br=y_br+offset_br[:,:,1]

        offset_tl=tf.reshape(offset_tl,(batch,k,1,2))
        offset_br=tf.reshape(offset_br,(batch,1,k,2))

        #all k boxes
        bboxes=tf.stack((x_tl,y_tl,x_br,y_br),axis=-1)

        tag_tl=map_to_vector(tag_tl,position_tl)
        tag_tl=tf.reshape(tag_tl,(batch,k,1))
        tag_br=map_to_vector(tags_br,position_br)
        tag_br=tf.reshape(tag_br,(batch,1,k))
        dists=tf.abs(tag_tl-tag_br)

        value_tl=expand_copy(value_tl,k,False)
        value_br=expand_copy(value_br,k,True)
        scores=(value_tl+value_br)/2
        invalid=-tf.ones_like(scores)

        class_tl=expand_copy(class_tl,k,False)#[batch,k,k]
        class_br=expand_copy(class_br,k,True)

        mask_scores=tf.where(tf.cast(tf.equal(class_tl,class_br),tf.int32)>0,scores,invalid)
        mask_scores=tf.where(dists<ae_threshold,mask_scores,invalid)
        mask_scores=tf.where(x_tl<x_br,mask_scores,invalid)
        mask_scores=tf.where(y_tl<y_br,mask_scores,invalid)

        mask_scores=tf.reshape(mask_scores,(batch,-1))
        scores,indexs=tf.nn.top_k(mask_scores,num_dets)
        scores=tf.expand_dims(scores,-1)

        bboxes=tf.reshape(bboxes,(batch,-1,4))
        bboxes=map_to_vector(bboxes,indexs,transpose=False)

        class_=tf.reshape(class_br,(batch,-1,1))
        class_=map_to_vector(class_,indexs,transpose=False)

        value_tl=tf.reshape(value_tl,(batch,-1,1))
        value_tl=map_to_vector(value_tl,indexs,transpose=False)

        value_br=tf.reshape(value_br,(batch,-1,1))
        value_br=map_to_vector(value_br,indexs,transpose=False)

        detection=tf.concat(-1,[bboxes,scores,value_tl,value_br,class_])
        return detection


















