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

""" Convolutional Neural Network Training Examples.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
"""

import tensorflow as tf
from hyper import *
from densenet import *
import numpy as np
import os, sys, time

#sys.path.append('../common/')
#from eso_utils import *
FLAGS = tf.app.flags.FLAGS
num_classes = 10

tf.app.flags.DEFINE_string('load_ckpt_path', './model/model.ckpt-', '''ckpt model''')
tf.app.flags.DEFINE_string('ckpt_path', './model/model.ckpt-100', '''ckpt model''')
tf.app.flags.DEFINE_string('model_id', '100', '''ckpt model''')

def tensor4d_transform_nchw2nhwc(tensor):
    #hwcn -> nchw
    return tensor.transpose((0, 2, 3, 1))

def tensor4d_transform_nhwc2nchw(tensor):
    #hwcn -> nchw
    return tensor.transpose((0, 3, 1, 2))

# Create model
def conv_net(x):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    x = add_inference(x)

    x = tf.layers.dense(x, num_classes, name="linear")
    return x

# Construct model
# logits = conv_net(X)

def test_model(img, ckpt_path):
    #tf.reset_default_graph()
    with tf.Graph().as_default():
#        x_tensor = tf.placeholder(tf.float32, shape=[None, 227, 227, 3], name = 'valid_x')
        x_tensor = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name = 'valid_x')
#        y_tensor = tf.placeholder(dtype = tf.int32, shape = [None], name = 'valid_y')

        logist = conv_net(x_tensor)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True
        all_vars = tf.global_variables()
        saver = tf.train.Saver()
        sess = tf.Session(config = config)
        saver.restore(sess, ckpt_path)

    s = time.time()

#    resized_img = resize_img(img, 227, 227)
#    img = np.asarray(resized_img, dtype = np.float32)
#    img = np.reshape(img, (1, 227, 227, 3))
#    print (img.shape)    
    label = [0]
    probabilities = sess.run([logist], feed_dict = {\
                x_tensor: img,
                })
    #return output of the last layer
    return probabilities

def inference(img):
    #load ckpt 
    ckpt_path = FLAGS.ckpt_path
    if not ckpt_path:
        print ("ckpt path is not right\n")
        return None 

    begin = time.time()
    #global_step = ckpt_path.split('/')[-1].split('-')[-1]
    #if global_step != (FLAGS.model_id):
    #    print ('ckpt error! {}\t{}'.format(global_step, FLAGS.model_id))
    #print ('Start use step: {} valid model'.format(global_step))

    predict = test_model(img, ckpt_path)
    end = time.time()
    print ('Consum time: {}'.format(end - begin))

    return predict


if __name__ == '__main__':
    input = open("../TF2Caffe/line_1x1x28x28.txt", "r");
    inputdata = input.read()
    nums = inputdata.strip().split()
    nums = [float(d) for d in nums]
    import numpy as np
    nums = np.array(nums)
#    nums_shape = nums.reshape(1, 3, 227, 227)
    nums_shape = nums.reshape(1, 1, 28, 28)

    tmp_nums = tensor4d_transform_nchw2nhwc(nums_shape)

    y = inference(tmp_nums)    
#    y = y.reshape(1,29,29,168)
#    print(y)
#    pdb.set_trace()
#    y_nums = tensor4d_transform_nhwc2nchw(y)
#    for(x) in y_nums.flatten():
#        print("%f" % x) 
    print ("predict is {}".format(y))
