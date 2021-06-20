import cv2
import numpy as np
from keras import applications

def pre_process_image_for_vgg(img,input_size):
    """
        Resizes the image to input of VGGInputSize specified in the config dictionary
        Normalises the image
        Reshapes the image to an array of images e.g. [[img],[img],..]

        Inputs:
            img: a numpy array or array of numpy arrays that represent an image
            input_size: a tuple of (width, height )
    """
    if type(img) == np.ndarray: # Single image 
        resized_img = cv2.resize(img,input_size,interpolation = cv2.INTER_AREA)
        normalised_image = applications.vgg16.preprocess_input(resized_img)
        reshaped_to_array_of_images = np.array([normalised_image])
        return reshaped_to_array_of_images
    elif type(img) == list: # list of images
        img_list = img
        resized_img_list = [cv2.resize(image,input_size,interpolation = cv2.INTER_AREA) for image in img_list]
        resized_img_array = np.array(resized_img_list)
        normalised_images_array = applications.vgg16.preprocess_input(resized_img_array)
        return normalised_images_array

def get_conversions_between_input_and_feature(pre_processed_input_image_shape,feature_map_shape):
    """
        Finds the scale and offset from the feature map (output) of the CNN classifier to the pre-processed input image of the CNN
        Finds the inverse, pre-processed input image to feature map

        Input:
            pre_processed_input_image_shape: The 3/4d shape of the pre-processed input image that is passed to the backbone CNN classifier

        Returns a dictionary of values to easily pass variables
    """
    # Find shapes of feature maps and input images to the classifier CNN
    assert len(pre_processed_input_image_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single image [height,width,channels]
    assert len(feature_map_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single feature map [height,width,channels]
    if len(pre_processed_input_image_shape) == 3:
        img_height, img_width, _ = pre_processed_input_image_shape
    elif len(pre_processed_input_image_shape) == 4:
        _, img_height, img_width, _ = pre_processed_input_image_shape

    if len(feature_map_shape) == 3:
        features_height, features_width, _ = feature_map_shape
    elif len(feature_map_shape) == 4:
        _, features_height, features_width, _ = feature_map_shape

    # Find mapping from features map (output of backbone_model.predict) back to the input image
    feature_to_input_x_scale = img_width / features_width
    feature_to_input_y_scale = img_height / features_height

    # Put anchor points in the centre of 
    feature_to_input_x_offset = feature_to_input_x_scale/2
    feature_to_input_y_offset = feature_to_input_y_scale/2

    # Store as dictionary
    feature_to_input = {
        "x_scale": feature_to_input_x_scale
        ,"y_scale": feature_to_input_y_scale
        ,"x_offset" : feature_to_input_x_offset
        ,"y_offset" : feature_to_input_y_offset
    }

    # Find conversions from input image to feature map (CNN output)
    input_to_feature_x_scale = 1/feature_to_input_x_scale
    input_to_feature_y_scale = 1/feature_to_input_y_scale
    input_to_feature_x_offset = -feature_to_input_x_offset
    input_to_feature_y_offset = -feature_to_input_y_offset

    # Store as dictionary
    input_to_feature = {
        "x_scale": input_to_feature_x_scale
        ,"y_scale": input_to_feature_y_scale
        ,"x_offset" : input_to_feature_x_offset
        ,"y_offset" : input_to_feature_y_offset
    }

    return feature_to_input, input_to_feature

def get_input_coordinates_of_anchor_points(feature_map_shape,feature_to_input):
    """
        Maps the CNN output (Feature map) coordinates on the pre-processed input image space to the backbone CNN 
        Returns the coordinates as a flattened list of dictionaries with the format {"x":x,"y":y}
    """
    assert len(feature_map_shape) in [3,4] # Either a 4d array with [:,height,width,channels] or just a single feature map [height,width,channels]

    if len(feature_map_shape) == 3:
        features_height, features_width, _ = feature_map_shape
    elif len(feature_map_shape) == 4:
        _, features_height, features_width, _ = feature_map_shape

    # For the feature map (x,y) determine the anchors on the input image (x,y) as array 
    feature_to_input_coords_x  = [int(x_feature*feature_to_input["x_scale"]+feature_to_input["x_offset"]) for x_feature in range(features_width)]
    feature_to_input_coords_y  = [int(y_feature*feature_to_input["y_scale"]+feature_to_input["y_offset"]) for y_feature in range(features_height)]
    coordinate_of_anchor_points = [{"x":x,"y":y} for x in feature_to_input_coords_x for y in feature_to_input_coords_y]

    return coordinate_of_anchor_points

def convert_feature_coords_to_input_b_box(anchor_point_x_coord,anchor_point_y_coord,feature_to_input):
    """Convert feature map coordinates to a bounding box in the input space in a dictionary """
    anchor_point_in_input_space = {
        "x_centre" : anchor_point_x_coord*feature_to_input["x_scale"] + feature_to_input["x_offset"]
        ,"y_centre" : anchor_point_y_coord*feature_to_input["y_scale"] + feature_to_input["y_offset"]
    }
    box_width = feature_to_input["x_scale"]
    box_height = feature_to_input["y_scale"]
    bounding_box = {
        "x1" : int(round(anchor_point_in_input_space["x_centre"] - box_width/2))
        ,"y1" : int(round(anchor_point_in_input_space["y_centre"] - box_height/2))
        ,"x2" : int(round(anchor_point_in_input_space["x_centre"] + box_width/2))
        ,"y2" : int(round(anchor_point_in_input_space["y_centre"] +  box_height/2))  
    }
    return bounding_box

def scale_bounding_box(bounding_box,scale):
    """Scales bounding box coords (in dict from {x1,y1,x2,y2}) by x and y given by sclae in dict form {x,y}"""
    scaled_bounding_box = {
        "x1" : int(round(bounding_box["x1"]*scale["x"]))
        ,"y1" : int(round(bounding_box["y1"]*scale["y"]))
        ,"x2" : int(round(bounding_box["x2"]*scale["x"]))
        ,"y2" : int(round(bounding_box["y2"]*scale["y"]))
    }
    return scaled_bounding_box