"""
    Author: Alexander Shellabear
    Email: alexshellabear@gmail.com

    Lessons Learnt
        1) Do not exceed the length of the image with the anchor boxes
        2) np.array.reshape() allows you to change the change of your array, in this case you can flatten your array using
            np.array.reshape(-1), this will let np choose the shape as long as it all adds up to np.array.shape
            np.array.reshape(-1,1) means the array will be flattened but each value will have it's own cell
            https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
        3) Array slicing with numpy or lists
            list[start:stop:step]
            list[:] = all
            list[0::2] = start at 0 and only return every 2nd item
        4) If you want to explicitely add a new axis with the list then you can use
            np.newaxis which explicitely adds a new axis
            only works for numpy arrays not lists
            https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
        5) using bounding box regression means that the output of the RPN is in the following
            deltas (slight movements) = [dx_point_c,dy_point_c,dw_in_log_space,dh_in_log_space]
            It is interesting that regression for width and height are in the log space
            https://datascience.stackexchange.com/questions/30557/how-does-the-bounding-box-regressor-work-in-fast-r-cnn
        6) Originally was really confused as to why the output of the Region Proposal Network RPN was not giving the number
            of rows output was different to the number of proposed bounding boxes. I accidentally passed the number of anchor
            boxes per anchor aka the number of potential boxes per anchor point.
            I was only passing the dimensions of all potential boxes per anchor point
            This is also including those boxes which fall outside of the image
        7) If something is too difficult to understand make the problem simple. Break it into smaller sub problems that can be 
            solved and then added together.
        8) To get the first data point out of a generator in python use the next function. For example...
            Getting the child files of a path next(os.walk(path))[1]  
        9) Use the .get(item_name,"default if there is no item_name key") function to get a key from the specified item_name. 
            If it does not exist instead of erroring out you can pass it a default function
        10) The input size does not need to be 224,224 when using a vgg16 model that is toppless.
            This will change the size of the output feature map
        11) You can have a custom loss function and still have a model checkpoint.
            Importantly you can check out what metrics your model has through print(model.metrics_names)
            Then just add val to the front
            https://stackoverflow.com/questions/43782409/how-to-use-modelcheckpoint-with-custom-metrics-in-keras 
        12) Can't use image generator and flip the feature map data because it is not representing an image but instead is representing features
        13) Easiest way to access a pixel is flipped from x y to y x in cv2 and np
            https://stackoverflow.com/questions/54549322/why-should-i-use-y-x-instead-of-x-y-to-access-a-pixel-in-opencv
        14) To increase the image size by adding images to the left and right of each other follow this post
            https://stackoverflow.com/questions/7589012/combining-two-images-with-opencv
            To stack vertically (img1 over img2): vis = np.concatenate((img1, img2), axis=0)
            To stack horizontally (img1 to the left of img2): vis = np.concatenate((img1, img2), axis=1)
        15) To get the output shape of a particular layer use model.layers[X].output_shape
        16) How to get a requirements.txt
            pip freeze > requirements.txt

"""
import cv2
from keras import applications
import numpy as np
import random
from ml_library import dataset_util
from ml_library.cust_model import custom_model

if __name__ == "__main__":
    print("starting...")

    # Set up config files, currently all relative file paths.
    config = { 
        "ProcessedDatasetFolder" : r".\2. Processed Datasets"
        ,"ModelsFolder" : r".\3. Models"
        ,"ModelInputSize" : (224,224)
        ,"VideoFiles" : {
            "ObjectVideoPath" : r"1. Binary Classifier Data\object.mp4"
            }
    }

    # Get vgg model without top to extract features
    config["BackboneClassifier"] = applications.VGG16(include_top=False,weights='imagenet')

    # Create dataset and save
    dataset = dataset_util.dataset_generator(config)
    dataset.convert_video_to_data(config["VideoFiles"]['ObjectVideoPath'])
    dataset.save_dataset(mode="append new")

    # Build and load model
    region_classifer_model = custom_model(config)

    # Convert data into ML format and train
    x_data, y_data = dataset.get_machine_formatted_dataset()
    region_classifer_model.train(x_data, y_data)

    # Predict on same dataset but truncate rows to reduce processing time
    random_selected_dataset = dataset.get_n_random_rows(20)
    input_images = np.array([row["HumanFormat"]["InputImage"] for row in random_selected_dataset])
    bounding_boxes =  region_classifer_model.predict(input_images)

    # Display results and compare against ground truth and predictions
    for img_index, img in enumerate(input_images):
        bounding_boxes_in_img = [box["BoundingBox"] for box in bounding_boxes if box["ImageIndex"] == img_index]
        for box in bounding_boxes_in_img:
            cv2.rectangle(img,(box["x1"],box["y1"]),(box["x2"],box["y2"]),(255,255,255),2)
        dataset_image_side_by_side = random_selected_dataset[img_index]["HumanFormat"]["AllImagesSideBySide"]

        # Display images
        cv2.imshow("Predicted Bounding Box",img)
        cv2.imshow("Datset All Images Side By Side",dataset_image_side_by_side)
        cv2.waitKey(1000)
    
    cv2.destroyAllWindows() # used to close the windows displaying images etc

    print("finished...")






 