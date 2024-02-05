import cv2
import numpy as np

from live_classes import object_detector
from live_classes import image_segmentor

from box_tracking import tracker

from handle_target import handle_target
from smarter_crop import smart_crop


class instance_segmentor:
    def __init__(self, object_model_path:str, segmentor_model_path:str, max_objects:int = 5,
                 object_score_threshold:float = 0.6):
        """class that does instance segmentation on the target, where the target is the person
        that is holding the template image"""
        
        # initialise both models
        self.detector = object_detector(model=object_model_path,max_results=max_objects,
                                        score_threshold=object_score_threshold)
        
        self.segmentor = image_segmentor(model=segmentor_model_path)

        self.object_tracker = tracker()

        self.target_ID = None
        self.target_centroid = None

        self.detected_frame = None

    def loop_function(self, image:np.uint8):
        """Returns an annotated image of the frame, including bounding boxes, and target indication"""
        current_frame = image.copy()
        (width, height, _) = current_frame.shape

        # get bounding boxes and draw all the boxes
        bounding_boxes = self.detector.loop_function(current_frame)
        self.detected_frame = self.detector.get_annotated_image()

        # track the objects that have been detected by the tracker
        objects, template_corners = self.object_tracker.update(bounding_boxes, current_frame)

        # get the target ID
        self.target_ID = self.object_tracker.get_target_ID()

        # target centroid is associated to the target ID, therefore if there is no target there is no centroid
        if self.target_ID is None:
            self.target_centroid = None

        # if there are any objects that are being tracked
        if objects:
            for objectID, bounding_boxes in objects.items():
                x,y,w,h = bounding_boxes
                cX = int(x + w/2)
                cY = int(y + h/2)

                text = f"ID {objectID}"

                # draw their bounding box centroids and print their ID
                cv2.putText(self.detected_frame, text, (cX - 10, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.circle(self.detected_frame, (cX, cY), 4, (0, 165, 255), -1 )

                # if the current object is the target
                if objectID == self.target_ID:
                    # mark them as the target
                    cv2.putText(self.detected_frame, "TARGET", (cX - 20, cY + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # crop only for the target object
                    newX, newY, newW, newH = smart_crop(x,y,w,h,height,width)
                    target_crop = current_frame[newY:newY+newH, newX:newX+newW,:]

                    target_mask = self.segmentor.loop_function(target_crop)

                    if target_mask is not None:
                        # returns the centre of mask
                        mask_centroid = handle_target(target_mask)

                        if mask_centroid is not None:
                            cv2.circle(target_mask, mask_centroid, 4, 0, -1)

                            # draw the centroid of the object onto the detected frame
                            self.target_centroid = (mask_centroid[0] + newX, mask_centroid[1] + newY)
                            cv2.circle(self.detected_frame, self.target_centroid, 
                                       4, (0, 0, 255), -1)

            return self.detected_frame

    def get_traget_centroid(self):
        return self.target_centroid