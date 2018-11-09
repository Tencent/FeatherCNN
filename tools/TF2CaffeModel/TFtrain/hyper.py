//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

## The following flags are related to save paths, tensorboard outputs and screen outputs

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.2, '''dropout keepprob''')
tf.app.flags.DEFINE_string('model_dir', '../model/white-nbi/v1.6', '''model to save''')
tf.app.flags.DEFINE_string('valid_model_dir', '../model/valid_v3.3/', '''model to save''')
tf.app.flags.DEFINE_integer('batch_size', 2, '''batch size''')
tf.app.flags.DEFINE_integer('epoch_num', 800, '''epoch num''')

tf.app.flags.DEFINE_integer('use_elu', 0, '''is used elu or relu''')
tf.app.flags.DEFINE_integer('use_bn', 1, '''is used batchnorm or not''')
tf.app.flags.DEFINE_integer('growth_k', 48, '''filter size''')
tf.app.flags.DEFINE_integer('nb_block', 2, '''how many ( dense block + Transition Layer)''')
tf.app.flags.DEFINE_float('l2_scale', 0.001, '''l2 loss scale''')
tf.app.flags.DEFINE_integer('gpu_id', 0, '''used gpu device id''')
tf.app.flags.DEFINE_integer('use_image_whiten', 1, '''use image whiten replace minus-mean-value''')
tf.app.flags.DEFINE_integer('use_random_noisy', 1, '''use image random noisy''')
tf.app.flags.DEFINE_integer('num_gpus', 2, '''how many gpus to use''')

tf.app.flags.DEFINE_integer('num_classes', 2, '''num classes''')

## The following flags define hyper-parameters regards training
tf.app.flags.DEFINE_string('train_dir', '/share/home/jtmeng/software/tensorflow/ILSVRC2012/kunming_nbi/train/', '''pos data dir''')
tf.app.flags.DEFINE_string('valid_dir', '/share/home/jtmeng/software/tensorflow/ILSVRC2012/kunming_nbi/valid/', '''pos data dir''')

tf.app.flags.DEFINE_float('init_lr', 0.0001, '''Initial learning rate''')
tf.app.flags.DEFINE_float('lr_decay_factor', 0.95, '''How much to decay the learning rate each
time''')
#tf.app.flags.DEFINE_float('pos_weight', 1., '''pos weight''')
tf.app.flags.DEFINE_float('center_loss_ratio', 0.003, '''pos weight''')


tf.app.flags.DEFINE_bool('is_use_ckpt', False, '''is load pre model''')


tf.app.flags.DEFINE_string('data_format', 'NHWC', 'image training format')
#tf.app.flags.DEFINE_string('data_format', 'NCHW', 'image training format')
tf.app.flags.DEFINE_integer('height', 227, 'input image height')
tf.app.flags.DEFINE_integer('width', 227, 'input image width')
tf.app.flags.DEFINE_integer('channel', 3, 'input image channel')
tf.app.flags.DEFINE_integer('num_threads', 16, "number of threads to process input data")
