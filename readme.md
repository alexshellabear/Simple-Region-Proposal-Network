# Project Name: #
Simple Region Proposal Network & Utilities

# Description: #
The most basic DIY implementation of Region Proposal Network (RPN) and utility functions to produce a visual example of how the RPN works.

In essence it predicts a bounding box of an object given an image but it's pretty inaccurate because it's a simple implementation.  

![Datagen to training and prediction]()

There are 3 key components
1) Dataset: Generate, store, export to Keras training format
2) RPN model: Build/load, train, predict
3) Utility functions: Convert between, input image, feature maps etc

# Environment Set Up: #
How to set up the right environment to run this python code

1) Make sure you have python installed, use the following link https://www.python.org/downloads/
2) How to get started with python https://www.python.org/about/gettingstarted/
3) How to install the right packages/modules from requirements.txt https://stackoverflow.com/questions/7225900/how-can-i-install-packages-using-pip-according-to-the-requirements-txt-file-from
4) Set the working directory to the same as simple_region_proposal.py and execute simple_region_proposal.py as main using the code below.

```sh
$ python3 simple_region_proposal.py
```

Okay you're now set with the right environment and can execute the program let's get this show on the road!

# Running the Current Program #
The program does the following.
1) Check if the dataset has been generated, if so load it
2) If dataset not generated then use generate it. The pipeline of datagen is shown below of how the input image is labelled and then the superimposed and scaled output matrix (aka feature map) is activated to show which anchor points are activated. In thise code there is only one anchor box per anchor point, it does not account for object scale and aspect ratio.
3) Save the dataset file in two formats, MachineFormat or HumanFormat as a dictionary for ease of debugging
4) Build the simple region proposal network
5) Start training, if there is an existing trained model load that instead. Otherwise train and then save it as you go
6) Run predictions and save outputs to bounding box
7) Loop through the dataset and predicted bounding boxes and show the comparison between training and predictions

# Extending the Code to Your Own Dataset and Projects #
The key piece of code you will need to change to use this program for your own purposes is the labelling tool. To make things simple I used ```get_red_box()```, this creates a mask of anything that is a certain shade of red. 

Note:
1) The dataset_util.py module only works on video files, could be extended to images
2) There is no labelling function but this can be developed for your own dataset.  
3) Does not account for a large scale object and does not use Non-maximum Suppression (NMS).

# Project Research #
All the things that were looked at as part of this project

1) [Step by step explanation of RPN + extra] - https://dongjk.github.io/code/object+detection/keras/2018/05/21/Faster_R-CNN_step_by_step,_Part_I.html
2) [vgg with top=false will only output the feature maps which is (7,7,512), other solutions will have different features produced] - https://github.com/keras-team/keras/issues/4465
3) [Understanding anchor boxes] - https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/
4) [Faster RCNN - how they calculate stride] - https://stats.stackexchange.com/questions/314823/how-is-the-stride-calculated-in-the-faster-rcnn-paper
5) [Good article on Faster RCNN explained] - https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8
6) [Indicating that Anchor boxes should be determine by ratio and scale ratio should be width:height of 1:2 1:1 2:1 scale should be 1 1/2 1/3] - https://keras.io/examples/vision/retinanet/
7) [Best explanation of anchor boxes] - https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html#:~:text=Anchor%20boxes%20are%20a%20set,sizes%20in%20your%20training%20datasets
8) [Summary of object detection history, interesting read] - https://dudeperf3ct.github.io/object/detection/2019/01/07/Mystery-of-Object-Detection/
9) [Mask RCNN Jupyter Notebook] - https://github.com/matterport/Mask_RCNN/blob/master/samples/coco/inspect_model.ipynb
10) [RPN in Python Keras which i'm trying to understand] - https://github.com/dongjk/faster_rcnn_keras/blob/master/RPN.py
11) [RPN implementation Keras Python] - https://github.com/you359/Keras-FasterRCNN/blob/master/keras_frcnn/data_generators.py
12) [RPN implementation Well Commented] - https://github.com/virgil81188/Region-Proposal-Network/tree/03025cde75c1d634b608c277e6aa40ccdb829693
13) [RPN Loss function clearly explained] - https://www.geeksforgeeks.org/faster-r-cnn-ml/
14) [RPN Developed in Keras Python Framework] - https://github.com/alexmagsam/keras-rpn