import os
import random
import cv2
import numpy as np
import pickle
try:
    from .utility_functions import pre_process_image_for_vgg
    from .utility_functions import get_conversions_between_input_and_feature
    from .utility_functions import get_input_coordinates_of_anchor_points
    from .utility_functions import convert_feature_coords_to_input_b_box
except:
    print("You should not be executing the program from this python file...")

class dataset_generator():
    """Handles all things to do with dataset generation, saving and loading."""
    dataset_base_name = "dataset.p"

    def __init__(self,config):
        """Stores config variables and loads dataset"""
        self.backbone_model = config["BackboneClassifier"]
        self.model_input_size = config["ModelInputSize"]
        self.processed_dataset_folder = config["ProcessedDatasetFolder"]
        self.dataset = []

        self.load_dataset()

    def load_dataset(self):
        """Checks to see if the dataset pickle file exists. If so load it"""
        dataset_full_path = f"{self.processed_dataset_folder}{os.sep}{self.dataset_base_name}"
        if os.path.exists(dataset_full_path):
            self.dataset = pickle.load(open(dataset_full_path,"rb"))

    def has_video_already_been_processed(self,video_file_path):
        """Checks if the video has already been processed"""
        already_processed_videos_set = set([row["Meta"]["VideoPath"] for row in self.dataset])
        return video_file_path in already_processed_videos_set
             
    def save_dataset(self,mode="append new"):
        """
            Saves the dataset as a pickle file
            Checks if the folder exists, if not create the folder
            Check if the dataset exists, if not save the dataset
            If the mode is override then it will replace the file
            if the mode is append then it will check if there are new video files processed and then append
                the new video files to the existing dataset

            Returns:
                True, if new data saved
                False, if no new data is saved
        """
        # Create processed dataset folder if not existent
        if not os.path.exists(self.processed_dataset_folder):
            os.mkdir(self.processed_dataset_folder)

        # if dataset has not been saved before save
        dataset_full_path = f"{self.processed_dataset_folder}{os.sep}{self.dataset_base_name}"
        if not os.path.exists(dataset_full_path):
            pickle.dump(self.dataset, open(dataset_full_path,"wb"))
            return True

        if mode == "override": # Replace file
            pickle.dump(self.dataset, open(dataset_full_path,"wb"))
            return True
        elif mode == "append new": # Check if file contents in there and if not then append
            saved_dataset = pickle.load( open(dataset_full_path,"rb"))
            saved_dataset_processed_videos_set = set([row["Meta"]["VideoPath"] for row in saved_dataset])
            current_dataset_processed_videos_set = set([row["Meta"]["VideoPath"] for row in self.dataset])
            videos_processed_but_not_in_saved_dataset = current_dataset_processed_videos_set - saved_dataset_processed_videos_set

            if len(videos_processed_but_not_in_saved_dataset) > 0:
                additional_rows_to_save = [row for row in self.dataset if row["Meta"]["VideoPath"] in videos_processed_but_not_in_saved_dataset]
                saved_dataset.extend(additional_rows_to_save)
                pickle.dump(saved_dataset, open(dataset_full_path,"wb"))
                return True
        return False

    def get_machine_formatted_dataset(self):
        """produces a dataset in a format which ML can train on split into x and y data"""
        x_data = np.array([row["MachineFormat"]["Input"][0] for row in self.dataset])
        y_data = np.array([row["MachineFormat"]["Output"][0] for row in self.dataset])
        return x_data, y_data

    def convert_video_to_data(self,video_file_path):
        """
            Converts a video into a dataset which can be read by a human or the ML model

            Checks if the video has already been processed
            Checks that the video file can be read
            resizes the image to suit the model
            gets a mask of the object - A1
            Gets the feature map by passing the input image through the CNN backbone, in this case vgg16
            Finds conversions between the input image space and feature map space
            checks whether or not an object was found
            goes through each anchor point, creates a mask for that box and finds the iou with the box and object mask
            gets highest iou and gets the coordinate of the anchor point
            Creates the ML output matrix
            Displays images for debugging
            Saves data in a list as an array in a human and machine readable format

            Assumption 1: There is only [0,1] objects in each frame
            Assumption 2: There is only one anchor box per anchor point
        """
        if self.has_video_already_been_processed(video_file_path):
            return None

        cap = cv2.VideoCapture(video_file_path)
        assert cap.isOpened() # can open file
        
        index = -1
        while True:
            returned_value, frame = cap.read()
            if not returned_value:
                print("Can't receive frame (potentially stream end or end of file?). Exiting ...")
                break
            index += 1

            resized_frame = cv2.resize(frame,self.model_input_size)
            
            final_mask, final_result, object_identified = get_red_box(resized_frame)

            prediction_ready_image = pre_process_image_for_vgg(frame,self.model_input_size)
            feature_map = self.backbone_model.predict(prediction_ready_image)

            feature_to_input, input_to_feature = get_conversions_between_input_and_feature(prediction_ready_image.shape,feature_map.shape)
            coordinates_of_anchor_boxes = get_input_coordinates_of_anchor_points(feature_map.shape,feature_to_input)

            anchor_point_overlay_display_img = final_result.copy()
            if object_identified == True:
                iou_list = []
                
                for coord in coordinates_of_anchor_boxes:
                    anchor_box_mask = self.create_anchor_box_mask_on_input(coord,feature_to_input,final_mask.shape)
                    iou_list.append(self.get_iou_from_masks(final_mask, anchor_box_mask))
                    
                    self.draw_anchor_point_and_boxes(anchor_point_overlay_display_img,coord,feature_to_input)

                matching_anchor_box_index = iou_list.index(max(iou_list))
                matching_coord = coordinates_of_anchor_boxes[matching_anchor_box_index]
                
                cv2.circle(anchor_point_overlay_display_img,(matching_coord["x"],matching_coord["y"]),3,(255,255,255))

                output_shape = (feature_map.shape[1],feature_map.shape[2],1)
                ground_truth_output = np.zeros(output_shape,dtype=np.float64)

                coord_in_f_map = self.convert_input_image_coord_to_feature_map(matching_coord,input_to_feature)
                ground_truth_output[coord_in_f_map['y'],coord_in_f_map['x']] = [1.0]

            else:
                ground_truth_output = np.zeros(output_shape,dtype=np.float64)
            
            debug_image = self.gen_debug_image_and_display(resized_frame,final_mask,final_result,anchor_point_overlay_display_img,coord_in_f_map,feature_to_input)
            self.dataset.append({ 
                "Meta": {
                    "VideoPath" : video_file_path
                    ,"FrameIndex" : index
                }
                ,"MachineFormat" : {
                    "Input" : feature_map
                    ,"Output" : np.array([ground_truth_output])
                }
                ,"HumanFormat" : { 
                    "InputImage" : resized_frame
                    ,"ObjectMask" : final_mask
                    ,"MatchedCoord" : matching_coord
                    ,"ObjectDetected" : object_identified
                    ,"AllImagesSideBySide" : debug_image
                }
                })

            print(f"[{index}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}] max iou={max(iou_list)}, coord {iou_list.index(max(iou_list))}")
        cv2.destroyAllWindows()

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def gen_debug_image_and_display(resized_frame,final_mask,final_result,anchor_point_overlay_display_img,coord_in_f_map,feature_to_input,wait_time_ms = 10):
        """
            For debugging purposes to display all the images and returns the concatenated result
            Convert everything to colour to be compatible to be concatenated
            Note 1: The final_mask is a 2d array that must be converted into a 3d array with 3 channels
        """

        white_colour_image = np.ones(resized_frame.shape,dtype=np.uint8) * 255
        final_mask_colour = cv2.bitwise_and(white_colour_image,white_colour_image,mask=final_mask)

        debug_image = np.concatenate((resized_frame, final_mask_colour), axis=1)
        debug_image = np.concatenate((debug_image, final_result), axis=1)
        debug_image = np.concatenate((debug_image, anchor_point_overlay_display_img), axis=1)

        ground_truth_bbox = convert_feature_coords_to_input_b_box(coord_in_f_map["x"],coord_in_f_map["y"],feature_to_input)
        ground_truth_output_colour = np.zeros(resized_frame.shape,dtype=np.uint8)
        cv2.rectangle(ground_truth_output_colour,(ground_truth_bbox["x1"],ground_truth_bbox["y1"]),(ground_truth_bbox["x2"],ground_truth_bbox["y2"]),(255,255,255),-1)

        debug_image = np.concatenate((debug_image, ground_truth_output_colour), axis=1)
        
        cv2.imshow("debug_image",debug_image)
        cv2.waitKey(wait_time_ms)
        return debug_image

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def draw_anchor_point_and_boxes(img_to_draw_on,anchor_point_coord,feature_to_input,scale=None,aspect_ratio=None):
        """For debugging to see the input image with the anchor points and boxes drawn and returns bounding box"""
        # TODO: Account for scale and aspect ratio

        cv2.circle(img_to_draw_on,(anchor_point_coord["x"],anchor_point_coord["y"]),2,(0,0,255))

        bounding_box = {
            "x1" : int(round(anchor_point_coord["x"] - feature_to_input["x_offset"]))
            ,"y1" : int(round(anchor_point_coord["y"] - feature_to_input["y_offset"]))
            ,"x2" : int(round(anchor_point_coord["x"] + feature_to_input["x_offset"]))
            ,"y2" : int(round(anchor_point_coord["y"] + feature_to_input["y_offset"]))
        }
        cv2.rectangle(img_to_draw_on,(bounding_box["x1"],bounding_box["y1"]),(bounding_box["x2"],bounding_box["y2"]),(255,255,255),1)

        return bounding_box

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def create_anchor_box_mask_on_input(coord,feature_to_input,mask_shape,scale=None,aspect_ratio=None):
        """Creates a mask where that covers the anchor box. This is used to then find the IOU of a blob"""
        # TODO: Account for scale and aspect ratio
        assert len(mask_shape) == 2 # Should be an image [width,height] because it is a mask

        # Create empty mask
        anchor_box_mask = np.zeros(mask_shape, dtype=np.uint8)
        
        x1 = int(round(coord["x"] - feature_to_input["x_offset"]))
        y1 = int(round(coord["y"] - feature_to_input["y_offset"]))
        x2 = int(round(coord["x"] + feature_to_input["x_offset"]))
        y2 = int(round(coord["y"] + feature_to_input["y_offset"]))

        fill_constant = -1
        anchor_box_mask = cv2.rectangle(anchor_box_mask,(x1,y1),(x2,y2),255,fill_constant)

        return anchor_box_mask

    #@staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def get_iou_from_masks(self,single_blob_mask_1, single_blob_mask_2): # TODO find variable
        """
            Gets the Intersection Over Area, aka how much they cross over divided by the total area
            from masks (greyscale images)

            Uses bitwise or to create new mask for union blob
            Uses bitwise and to create new mask for intersection blob

            Assumption 1: The masks must be greyscale
            Assumption 2: There must only be one blob (aka object) in each mask
            Assumption 3: Both masks must be the same dimensions (aka same sized object)
            Note 1: If the union area is 0, there are no blobs hence the IOU should be 0
        """
        assert len(single_blob_mask_1.shape) == 2 # Should be a greyscale image
        assert len(self.get_area_of_blobs(single_blob_mask_1)) == 1 # Mask should only have one blob in it
        assert len(single_blob_mask_2.shape) == 2 # Should be a greyscale image
        assert len(self.get_area_of_blobs(single_blob_mask_2)) == 1 # Mask should only have one blob in it
        assert single_blob_mask_1.shape[0] == single_blob_mask_2.shape[0] and single_blob_mask_1.shape[1] == single_blob_mask_2.shape[1]

        union_mask = cv2.bitwise_or(single_blob_mask_1,single_blob_mask_2)
        if len(self.get_area_of_blobs(union_mask)) == 1:
            union_area = self.get_area_of_blobs(union_mask)[0]
        else: 
            intersection_over_union = 0.0 # Stop math error, divide by 0
            return intersection_over_union

        intersection_mask = cv2.bitwise_and(single_blob_mask_1,single_blob_mask_2)
        if len(self.get_area_of_blobs(intersection_mask)) == 1:
            intersection_area = self.get_area_of_blobs(intersection_mask)[0]
        else: 
            intersection_area = 0.0

        intersection_over_union = intersection_area / union_area
        assert intersection_over_union >= 0.0
        assert intersection_over_union <= 1.0

        return intersection_over_union

    @staticmethod
    def get_area_of_blobs(mask):
        """
            Takes a cv2 mask, converts it to blobs and then finds the area and returns the blobs and the corresponding area
        """
        contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blob_areas = [cv2.contourArea(blob) for blob in contours]
        return blob_areas

    @staticmethod # Call this function when you don't initialise the class, means you don't pass the self variable
    def convert_input_image_coord_to_feature_map(coord_in_input_space,input_to_feature):
        """
            Converts an point in the input image space to the feature map space returns as a dictionary

            Assumption 1: coord_in_input_space is a dictionary of {"x",int,"y",int}
            Assumption 2: input_to_features is a dictionary
        """
        x = int(round((coord_in_input_space["x"] + input_to_feature["x_offset"])*input_to_feature["x_scale"]))
        y = int(round((coord_in_input_space["y"] + input_to_feature["y_offset"])*input_to_feature["y_scale"]))
        coord_in_feature_map = {"x":x,"y":y}
        return coord_in_feature_map

    def get_n_random_rows(self,n_random_rows):
        """Gets n random rows from the dataset"""
        return random.sample(self.dataset, n_random_rows)



def get_red_box(resized_frame,threshold_area = 400):
    """
        Uses HSV colour space to determine if a colour is actually red.
            Does this by considering the lower and upper colour space.
            Adds those two masks together
            Uses morphology to fill in the small gaps
            finds those blobs that have an area greater than threshold_area
            returns the overlayed image and the mask with only blobs greater than the threshold_area

            TODO refeactor code with new general utility functions, store away in other module for better use next timme
    """
    hsv_colour_img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2HSV)
    #lower red
    lower_red = np.array([0,110,110])
    upper_red = np.array([10,255,255])

    #upper red
    lower_red2 = np.array([170,50,50])
    upper_red2 = np.array([180,255,255])

    mask = cv2.inRange(hsv_colour_img, lower_red, upper_red)
    mask2 = cv2.inRange(hsv_colour_img, lower_red2, upper_red2)

    combined_mask = cv2.bitwise_or(mask,mask2)
    
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    morphed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    final_mask = mask = np.zeros(mask.shape, dtype=np.uint8)

    contours, _  = cv2.findContours(morphed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs_greater_than_threshold = [blob for blob in contours if cv2.contourArea(blob) > threshold_area]
    for blob in blobs_greater_than_threshold:
        cv2.drawContours(final_mask, [blob], -1, (255), -1)

    final_result = cv2.bitwise_and(resized_frame,resized_frame, mask= final_mask)

    object_identified = final_mask.max() > 0

    return final_mask, final_result, object_identified




