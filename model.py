import os
import numpy as np
import tensorflow as tf
from config import CLASS_NUM, IMAGE_SIZE, TRAIN_MODE_IN_TRAIN, TRAIN_MODE_IN_TEST, TRAIN_MODE_IN_VALID, IMAGE_MODE, DATA_FORMAT, ACTIVATION
from decision_head import decision_head
from tensorflow.python import pywrap_tensorflow
slim=tf.contrib.slim


class Model(object):

    def __init__(self, sess, param):
        self.step = 0
        self.session = sess
        self.__learn_rate = param["learn_rate"]
        self.__max_to_keep = param["max_to_keep"]
        self.__pb_model_path = param["pb_Mode_dir"]
        self.__restore = param["b_restore"]
        self.__bn_momentum = param["momentum"]
        self.__mode = param["mode"]
        # self.__mode = "testing"
        self.backbone = param["backbone"]
        self.tensorboard_logdir = param["tensorboard_logdir"]
        self.neck = param["neck"]
        if param["mode"] == 'testing' or param["mode"] == 'savePb':
            self.is_training = TRAIN_MODE_IN_TEST
        else:
            self.is_training = TRAIN_MODE_IN_TRAIN
        if param["mode"] == "train_segmentation":
            self.keep_dropout_backbone = True
            self.keep_dropout_head = True
        elif param["mode"] == "train_decision":
            self.keep_dropout_backbone = False
            self.keep_dropout_head = True
        else:
            self.keep_dropout_backbone = False
            self.keep_dropout_head = False
        self.__batch_size = param["batch_size"]
        self.__batch_size_inference = param["batch_size_inference"]

        if param["mode"] == 'savePb' or param["mode"] == 'visualization' or param["mode"] == 'testing':
            self.is_pb = True
        else:
            self.is_pb = False

        # Building graph
        with self.session.as_default():
            self.build_model()
        # 参数初始化，或者读入参数
        with self.session.as_default():
            self.init_op.run()
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
            var_list += bn_moving_vars
            self.__saver = tf.train.Saver(var_list, max_to_keep=self.__max_to_keep)
            # Loading last save if needed
            if self.__restore:
                ckpt = tf.train.latest_checkpoint(self.__checkPoint_dir)
                if ckpt:

                    var_list2 = [v for v in var_list if "bias" not in v.name]

                    self.__saver2 = tf.train.Saver(var_list2, max_to_keep=self.__max_to_keep)

                    self.step = int(ckpt.split('-')[1])
                    self.__saver2.restore(self.session, ckpt)
                    print('Restoring from epoch:{}'.format(self.step))
                    self.step += 1

                    if self.__mode == 'savePb':
                        reader = pywrap_tensorflow.NewCheckpointReader(ckpt)
                        var_to_shape_map = reader.get_variable_to_shape_map()
                        source_list = [key for key in var_to_shape_map if "CBR" in key]
                        epsilon = 0.001
                        for key in source_list:
                            if "moving_mean" in key:
                                mean = np.array(reader.get_tensor(key))

                                key_var = key[0:-11] + "moving_variance"
                                var = np.array(reader.get_tensor(key_var))

                                key_gamma = key[0:-11] + "gamma"
                                gamma = np.array(reader.get_tensor(key_gamma))

                                key_beta = key[0:-11] + "beta"
                                beta = np.array(reader.get_tensor(key_beta))

                                key_W = key[0:-14] + "Conv2D/kernel"
                                W = np.array(reader.get_tensor(key_W))

                                alpha = gamma / ((var + epsilon) ** 0.5)

                                W_new = W * alpha

                                B_new = beta - mean * alpha

                                weight = tf.get_default_graph().get_tensor_by_name(key_W + ':0')

                                update_weight = tf.assign(weight, W_new)

                                bias_name = key_W[0:-6] + 'bias:0'

                                bias = tf.get_default_graph().get_tensor_by_name(bias_name)

                                updata_bias = tf.assign(bias, B_new)

                                sess.run(update_weight)
                                sess.run(updata_bias)

    def build_model(self):

        # tf.summary.image('input_image', image_input, 10)
        # tf.summary.image('mask', mask, 10)

        if (self.is_pb == False):
            is_training_seg = tf.placeholder(tf.bool, name='is_training_seg')
            is_training_dec = tf.placeholder(tf.bool, name='is_training_dec')
            if IMAGE_MODE == 0:
                image_input = tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
                                             name='Image')
                num_ch = 1
            else:
                image_input = tf.placeholder(tf.float32, shape=(self.__batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                             name='Image')
                num_ch = 3
            label = tf.placeholder(tf.float32, shape=(self.__batch_size, CLASS_NUM), name='Label')

        else:
            is_training_seg = False
            is_training_dec = False
            if IMAGE_MODE == 0:
                image_input = tf.placeholder(tf.float32,
                                             shape=(self.__batch_size_inference, IMAGE_SIZE[0], IMAGE_SIZE[1], 1),
                                             name='Image')
                num_ch = 1
            else:
                image_input = tf.placeholder(tf.float32,
                                             shape=(self.__batch_size_inference, IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                                             name='Image')
                num_ch = 3
            label = tf.placeholder(tf.float32, shape=(self.__batch_size_inference, CLASS_NUM), name='Label')



        from GhostNet import ghostnet_base
        self.__checkPoint_dir = 'checkpoint/ghostnet'
        # Set depth_multiplier to change the depth of GhostNet
        backbone_output = ghostnet_base(image_input, mode=self.__mode, data_format=DATA_FORMAT, scope='segmentation',
                                        dw_code=None, ratio_code=None,
                                        se=1, min_depth=8, depth=1, depth_multiplier=1, conv_defs=None,
                                        is_training=is_training_seg, momentum=self.__bn_momentum)



        # Create decision head
        dec_out = decision_head(backbone_output[-1], backbone_output[1], class_num=1024, scope='decision',
                                keep_dropout_head=self.keep_dropout_head,
                                training=is_training_dec, data_format=DATA_FORMAT, momentum=self.__bn_momentum,
                                mode=self.__mode, activation=ACTIVATION)
        dec_out = tf.nn.sigmoid(dec_out)
        dec_out = slim.fully_connected(dec_out, CLASS_NUM, activation_fn=None)
        decision_out = tf.nn.sigmoid(dec_out)
        decision_out = tf.argmax(decision_out, axis=1, name='decision_out')
        decision_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=dec_out, labels=label)
        decision_loss = tf.reduce_mean(decision_loss)

        if self.is_pb == False:
            # Variable list
            train_decision_var_list = [v for v in tf.trainable_variables()]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            update_ops_decision = [v for v in update_ops]


            optimizer_decision = tf.train.AdamOptimizer(self.__learn_rate)


            with tf.control_dependencies(update_ops_decision):
                optimize_decision = optimizer_decision.minimize(decision_loss, var_list=train_decision_var_list)

            self.decision_loss = decision_loss
            self.optimize_decision = optimize_decision

        if not os.path.exists(self.tensorboard_logdir):
            os.makedirs(self.tensorboard_logdir)

        init_op = tf.global_variables_initializer()


        self.Image = image_input
        self.is_training_seg = is_training_seg
        self.is_training_dec = is_training_dec
        self.label = label
        self.decison_out = decision_out
        self.init_op = init_op


    def save(self):
        if not os.path.exists(self.__checkPoint_dir):
            os.makedirs(self.__checkPoint_dir)

        self.__saver.save(
            self.session,
            os.path.join(self.__checkPoint_dir, 'ckp'),
            global_step=self.step
        )


    def save_PbModel(self):
        input_name = "Image"
        output_name = "decision_out"
        output_node_names = [input_name, output_name]
        # output_node_names = [input_name]
        print("模型保存为pb格式")
        output_graph_def = tf.graph_util.convert_variables_to_constants(self.session,
                                                                        self.session.graph_def,
                                                                        output_node_names)
        output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def, protected_nodes=None)

        if not os.path.exists(self.__pb_model_path):
            os.makedirs(self.__pb_model_path)
        pbpath = os.path.join(self.__pb_model_path, 'frozen_inference_graph_fuse_carrier.pb')
        with tf.gfile.GFile(pbpath, mode='wb') as f:
            f.write(output_graph_def.SerializeToString())



    def freeze_session(self, keep_var_names=None, output_names=["decision_out"], clear_devices=True):
        from tensorflow.python.framework.graph_util import convert_variables_to_constants
        session = self.session
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                          output_names, freeze_var_names)
            return frozen_graph
