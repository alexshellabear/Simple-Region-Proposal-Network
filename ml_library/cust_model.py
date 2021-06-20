import os
from keras import layers
from keras import Model
from keras import losses
from keras import utils
from keras.callbacks import ModelCheckpoint
import datetime
from keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2
try:
    from .utility_functions import get_conversions_between_input_and_feature
    from .utility_functions import get_input_coordinates_of_anchor_points
    from .utility_functions import convert_feature_coords_to_input_b_box
    from .utility_functions import scale_bounding_box
except:
    print("You should not be executing the program from this python file...")

class custom_model():
    """Handles all things to do with building, training, saving, loading and predicting"""
    model_name = "region_classifier"

    def __init__(self,config):
        self.backbone_classifier = config["BackboneClassifier"]
        self.backbone_classifier_input_shape = config["ModelInputSize"]
        self.backbone_output_channels = config["BackboneClassifier"].layers[-1].output_shape[-1]
        self.history = None

        self.models_folder = config["ModelsFolder"]
        if not os.path.exists(self.models_folder):
            os.mkdir(self.models_folder)
        
        self.build_region_classifier()

        self.load_model()

    def predict(self,input_array,threshold=0.9):
        """
            Takes an input image and predicts where the bounding box is for that image 
            Methodology:
                Gets dimensions of array
                Checks if the array is ML ready (scaled, normalised)
                passes through backbone classifier to get featuremap
                Run predictions on feature map
                Convert coordinates from feature map to scaled size
                Convert coordinates from scaled size to input size
            Input:
                input_array CV2 colour image, normalised input image or feature map
                threshold a float to determine the minimum confidence to accumulate data
            Output:
                List of dictionaries in the following format
                {
                    "ImageIndex": The index of the multiple images passed. If only one image in 3d array passed, will be 0
                    ,"BoundingBox" : {x1,y1,x2,y2}
                }

            Note:
                Can accept and array of multiple images and hence provide a dictionary o
            
        """
        num_dimensions = len(input_array.shape)
        assert num_dimensions in [3,4] # must be a colour image in cv2 format or in ML ready array
        if num_dimensions == 3:
            input_array = np.array([input_array])
        _, height, width, channels = input_array.shape
        assert channels == 3 # must be a colour image

        # Scale input image to meet input size of backbone classifier
        if height != self.backbone_classifier_input_shape[0] and width != self.backbone_classifier_input_shape[1]:
            scale_feature_to_input = {
                "x" : width / self.backbone_classifier_input_shape[1]
                ,"y" : height / self.backbone_classifier_input_shape[0]
            } 
            input_array = np.array([cv2.resize(input_array[0],self.backbone_classifier_input_shape ,interpolation = cv2.INTER_AREA)])
        else:
            scale_feature_to_input = {
                "x" : 1.0
                ,"y" : 1.0
            } 

        if not custom_model.is_array_pre_processed(input_array):
            input_array = preprocess_input(input_array)
        
        # Perform predictions of object location
        feature_map_array = self.backbone_classifier.predict(input_array)
        feature_map_pixels_prop_of_object_array = self.model.predict(feature_map_array)

        # Get conversions from feature map to input array
        feature_to_input, input_to_feature = get_conversions_between_input_and_feature(input_array.shape,feature_map_array.shape)

        # Find anchor points that exceed threshold
        # TODO have scale and aspect ratio
        loc_of_feature_map_exceed_thresh_array = np.argwhere(feature_map_pixels_prop_of_object_array >= threshold)

        # Convert feature map anchor coordiantes to a bounding box in the input space
        bounding_boxes_and_img_indices = [{
            "ImageIndex":img_index
            ,"BoundingBox" : convert_feature_coords_to_input_b_box(height,width,feature_to_input)
         } for img_index, width,height,_ in loc_of_feature_map_exceed_thresh_array]

        # Scale the bounding boxes to accommodate for scale down to meet backbone classifier required input size
        scaled_bounding_boxes_and_img_indicies = [{
            "ImageIndex":box_and_index["ImageIndex"]
            ,"BoundingBox" : scale_bounding_box(box_and_index["BoundingBox"],scale_feature_to_input)
         } for box_and_index in bounding_boxes_and_img_indices]

        return scaled_bounding_boxes_and_img_indicies

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def is_array_pre_processed(input_array):
        """Determines if the array has been pre-processed by assuming a uint8 is not processed and float is"""
        return "float" in str(input_array.dtype)

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def is_array_normalised(input_array):
        """Determines if the array has been normalised i.e between 0.0 - 1.0"""
        max_val = input_array.max()
        min_val = input_array.min()
        return max_val >= 0.0 and max_val <= 1.0 and min_val >= 0.0 and min_val <= 1.0

    def load_model(self,mode="most_recent"):
        """Loads the model based upon the latest training session and min loss score"""
        base_folder, all_training_folders, _  = next(os.walk(self.models_folder))
        if len(all_training_folders) == 0:
            return False
        datetime_list = [datetime.datetime.strptime(folder, "%d %B %Y %Hhrs %Mmins") for folder in all_training_folders]
        max_datetime = max(datetime_list)
        max_index = datetime_list.index(max_datetime)
        most_recent_folder_path = f"{base_folder}{os.sep}{all_training_folders[max_index]}"

        base_folder, _ , model_files_in_most_recent_folder = next(os.walk(most_recent_folder_path))
        if len(model_files_in_most_recent_folder) == 0:
            return False
        # Lambda functions for readability to get accuracy as float
        get_base_name = lambda full_file_name : os.path.splitext(full_file_name)[0]
        get_loss = lambda base_name : float(base_name.split("-")[-1])
        loss_list = [get_loss(get_base_name(f)) for f in model_files_in_most_recent_folder] 
        min_loss = min(loss_list)
        min_index = loss_list.index(min_loss)

        path_most_recent_lowest_loss_model = f"{base_folder}{os.sep}{model_files_in_most_recent_folder[min_index]}"
        self.model.load_weights(path_most_recent_lowest_loss_model)

        print(f"{self.model_name}: Loaded model from {path_most_recent_lowest_loss_model}")
        return True

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def get_todays_datetime_as_string():
        """Get todays date as string in the form 11 June 2021 17hrs 5Minutes"""
        return datetime.datetime.today().strftime("%d %B %Y %Hhrs %Mmins")

    def build_region_classifier(self):
        """Creates the RPN classifier model"""
        assert type(self.backbone_output_channels) == int # Should be an integer

        # Input layer, should have the same number of channels that the headless classifier produces
        feature_map_input = layers.Input(shape=(None,None,self.backbone_output_channels),name="RPN_Input_Same")
        # CNN component, ensure that padding is the same so that it has the same dimensions as input feature map
        convolution_3x3 = layers.Conv2D(filters=self.backbone_output_channels,kernel_size=(3, 3),name="3x3",padding="same")(feature_map_input)
        # Output objectivness
        objectivness_output_scores = layers.Conv2D(filters=1, kernel_size=(1, 1),activation="sigmoid",kernel_initializer="uniform",name="scores1")(convolution_3x3)
        # Create model with input feature map and output
        self.model = Model(inputs=[feature_map_input], outputs=[objectivness_output_scores])
        # Set loss and compile
        self.model.compile(optimizer='adam', loss={'scores1':losses.binary_crossentropy})

    def train(self,x_data,y_data,mode="use loaded model"):
        """Trains and saves the best model""" 
        if self.model == None and mode == "use loaded model":
            # Create folder to save models
            training_folder = f"{self.models_folder}{os.sep}{self.get_todays_datetime_as_string()}"
            if not os.path.exists(training_folder):
                os.mkdir(training_folder)
            file_path = f"{training_folder}{os.sep}{self.model_name}"

            check_point = ModelCheckpoint(filepath=file_path+"-{epoch:02d}-{val_loss:.3f}.hdf5",
                                            monitor="val_loss",
                                            mode="min",
                                            save_best_only=True,
                                            )

            self.history = self.model.fit(x=x_data,y=y_data, batch_size=8, epochs=30, verbose=1,validation_split=0.1,callbacks=[check_point])
        else:
            print(f"{self.model_name}: model already loaded skip training")
if __name__ == "__main__":
    print("Probably shouldn't be executing this file as a main function")