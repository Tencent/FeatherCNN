# TF2CaffeModel

## Introduction

TF2CaffeModel is a simple model converter which can convert TensorFlow model to Caffe model using customized Tensorlayer.

## Enviroment Setting

There is a minmum version requirement on Python, Tensorflow, CUDA, and CUDNN. A recommanded enviroment to deploy our model converter needs CUDA 9.0, CUDNN 7.0, TensorFlow 1.8.0 and above, and Python 3. 

## Model definition with customized Tensorlayer.
Based on Tensorlayer, we extend several commonly used layers with net struture and weight dump function for caffe. These layers include Conv, Batch_normlization, Scale, Average_pooling, Max_pooling, Concat, Global_average_pooling, Dropout layers. These layers can be used to reformulate most commonly used networks. Other specail layers and customer-definited layers which are not currently supported, must be extend in our customized Tensorlayer (ITL) first with structure and weight dump functions for caffe, and use it afterwords. 

The following layers have been redefinited with caffe support, please check before use it.  

    def conv_layer(self, inputs, filters, kernel, stride=1, layer_name="conv"):
        kernel_size = kernel[0]
        network = tl.layers.Conv2d(inputs, filters, kernel, strides=(stride,stride), act=None, name=layer_name)
        return network

    def global_average_pooling(self, inputs):
        network = tl.layers.GlobalMeanPool2d(inputs)
        return network

    def batch_normlization(self, inputs, layer_name="batchnorm"):
        network = tl.layers.BatchNormLayer(inputs, decay=0.9, act=tf.nn.relu, is_train=self.is_training, name=layer_name)
        return network

    def dropout(self, inputs, rate, layer_name="dropout"):
        return inputs

    def average_pooling(self, inputs, pool_size=[2, 2], strides=2, padding='VALID', layer_name="pool"):
        network = tl.layers.MeanPool2d(inputs, filter_size=pool_size, strides=strides, padding=padding, name=layer_name)
        return network

    def concat(self, inputs, layer_name="concat"):
        network = tl.layers.ConcatLayer(inputs, self.channel_index, name=layer_name)
        return network

    def linear(self, inputs, units):
        return tf.layers.dense(inputs=inputs, units=units, name="linear")

    def max_pooling(self, inputs, pool_size=[3,3], stride=2, padding='VALID', layer_name="max_pool"):
        network = tl.layers.MaxPool2d(inputs, pool_size, strides=(stride, stride), padding=padding, name=layer_name)
        return network

There are several network examples in directory "models" for you to use directly, modify or extend to your own networks. 

## Trainning with customized Model 
  hyper.py is used to set the path of the trainning dataset, trained models and etc, please check that file for more details. 
  We can use '''python3 train.py''' to start the training progress. 


## Inferencing with customized Model
  We can use '''python model_inference.py''' to start the inference test exampls. The time usage on inferencing with the given model will be presented on the terminal. You can also output the value of the tensor in the final layer by modifying the folowing code and enable the following two lines:

\#    for(x) in y_nums.flatten():
\#        print("%f" % x) 

## Model Convert from TF to Caffe
   1. Construct network structure from TF 
   We first rewrite the targe model with ITL, then run this program to generate network structure and store it into a protobuf file. There is an example in net.py demonstrated the code of rewriting densenet.py and generated the equivalent model for densenet. (You have to delete all the heading lines manually descriping the information of tensors until the line start with "name: ....")  

   2. Extract the weights from TF model.
   With the above network structure (a protobuf file), please specify the following parameters for model checkpoint path, prototxt, and model_name. Then run 2nd_tf2caffe.py to collect all weights from tensorflow's checkpoint data and rearrage them into caffemodel. 
   

    tf.app.flags.DEFINE_boolean('use_tfplus', False, 'tfplus mode')
    tf.app.flags.DEFINE_string('model_checkpoint_path', '../model/white-nbi/v1.6/', 'the path of the saved model')
    tf.app.flags.DEFINE_string('prototxt', './tran7.prototxt', 'the net definition file')
    tf.app.flags.DEFINE_string('model_name', './tran7', 'the model name')


## Model Results Evaluation with Caffe
   Before use the generated prototxt and caffemodel with your caffe in your production, it is recommanded to double check the results of caffe with Tensorflow. 
   For tensorflow, model_inferece.py can be used to genereate the results for a given input file, such as line_3_227_227.txt. Then we can use it to check the results computed by caffe with our generated model. 




