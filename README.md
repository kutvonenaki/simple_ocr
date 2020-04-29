# Simple digit OCR with tf.keras 2.1

Related blog post:  

A simple CRNN based OCR and inference to recognize digits. Training done using generated digits from https://textrecognitiondatagenerator.readthedocs.io/en/latest/index.html. The code here is meant to serve as an easy as possible end-to-end example from data generation to inference, or as a template for further more complicated OCR implementations. The focus has been on the CRNN code architecture and not on the neural network architecture itself. User can later tune the NN architecture to their liking. The data and model is simple enough so that the training can be done on CPU.

There are some other simple implementations out there such as  
https://keras.io/examples/image_ocr/  
Here we have made the code tf.keras 2.1 compatible, emphasis has been on as clear as possible construction of the components and make the code readable and beginner friendly with lot's of comments in the code. Some implementations also require to define the image width which is not really needed in the NN model. Here the width is set batch by batch based on the largest image in the batch in the batch data generator. Also in our implementation user doesn't need to define absolute maximum length for strings, the length is dynamically setted based on the longest word in a batch. Also there's a inference function for easy testing on external data.

## How to
Open OCR_simple notebook and try. Or you can run python train.py for training and python inference.py for inference but you won't see visual outputs.

## Note about the implementation

One of the trickier parts of CRNN in keras framework is that the output dimensions of the NN is of form (batch_size, time distributed dimension, number of unique characters in data) and the actual ground truth is the labels in the word, e.g. (1,3,6,2,9). Thus the prediction and the ground truth are not of same shape (one of the restrictions in tf.keras 2.1 simple loss functions, y_true and y_pred must match). In addition the CTC loss and decoding requires some additional information to ensure correct behaviour for different lenghts of input. So some magic is needed. Here we embedd the labels, input lengths etc inside array of shape (batch_size, time_dist_length, num_uniq_chars) and return that as y_true. Later the tensors are unpacked in the loss function.


## Installation

### TL;DR, Tensorflow 2.1, pip install trdg

### Those needing more support on setting up things with Docker:
Prerequirements
Linux for gpu tensorflow image, if want to use windows from docker, then drop the gpu tag and do training without gpu.
Or another option is to just use your conda environment on Windows with gpu support. In that case skip the docker parts.
Or make an environment by other means. Below instructions for the docker environment setup:  

More info:  
https://www.tensorflow.org/install/docker  

pull latest docker tensorlow image  
docker pull tensorflow/tensorflow:latest-gpu-py3-jupyter  

go the the OCR_simple folder:  
run the container mounting the current dir to /tf/work  
docker run -it --rm --gpus all -v  $PWD:/tf/work -p 8888:8888 tensorflow/tensorflow:latest-gpu-py3-jupyter  

Install some libraries which are not part of the docker container but required (run from notebook cell)
!apt-get update  
!apt-get install -y libsm6 libxext6 libxrender-dev

install the base text generator (run below from notebook cell)
!pip install trdg 

Restart jupyter kernel

Save the new docker image so that you don't need to do the above again  
open a new terminal and type docker ps to see the running containers  
check the CONTAINER ID corresponding to tensorflow/tensorflow tensorflow/tensorflow:latest-gpu-py3-jupyter,
for example 55d87004c378   

commit to a new image  
docker commit 55d87004c378 tensorflow:simple_ocr  

check that the image is created  
docker images  

Next time starting the container replace use  
docker run -it --rm --gpus all -v $PWD:/tf/work -p 8888:8888 tensorflow:simple_ocr 

## Play around with with trdg
list of possible parameters for trdg can be found at   
https://github.com/Belval/TextRecognitionDataGenerator/blob/master/trdg/generators/from_random.py  

to test generater images in jupyter notebook: 

from trdg.generators import GeneratorFromRandom  
base_generator = GeneratorFromRandom(use_symbols=False, use_letters=False)
img, lbl = next(base_generator)
display(img)


## Possible errors 

If you get error like: 
--> 117             y_true[:, 0:max_word_len_batch, 0] = batch_labels
    118             y_true[:, 0, 1] = label_lens
    119             y_true[:, 0, 2] = input_length

ValueError: could not broadcast input array from shape (12,7) into shape (12,6)  
Means that you're using too many maxpoolings or others means of dim reduction and the time dimension can not capture 
all the craracters. In another words, time distributed length 6 can not predict 7 characters.