import numpy as np
import cv2
from scipy.spatial import distance as dist 
from collections import OrderedDict
from template_matching import template_match
from template_matching import sift_bfmatch
from template_matching import orb_bfmatch
from template_matching import orb_flannmatch
from template_matching import brisk_flannmatch

template_image = cv2.imread("images\cat_image_2.jpg")

class tracker:
    def __init__(self, maxDisappeared = 50):
        """Takes in the maximum number of frames to be considered disappeared"""
        
        self.Objects = OrderedDict()    # stores the bounding boxes of each object
        self.disappeared = OrderedDict()# stores how many frames each object has disappeared
        self.nextObjectID = 0

        # keeps the tracking of the target
        self.target_disappeared = 0

        # key of the target
        self.TargetID = None

        self.maxDisappeared = maxDisappeared

    def get_target_ID(self):
        return self.TargetID

    def register(self, bounding_box):
        """Registers a new object with the next ID """
        self.Objects[self.nextObjectID] = bounding_box
        self.disappeared[self.nextObjectID] = 0

        # increment the next object ID
        self.nextObjectID += 1


    def unregister(self, objectID):
        """removes that object from the dictionary"""
        del self.Objects[objectID]
        del self.disappeared[objectID]

        # set the target ID to none if the target is lost
        if objectID == self.TargetID:
            self.TargetID = None

    def calculate_centroids(bounding_boxes):
        """calculates the centres of each bounding box"""
        centroids = np.zeros((len(bounding_boxes), 2), dtype = "int")

        for count, (x, y, w, h) in enumerate(bounding_boxes):
            cX = x + w/2
            cY = y + h/2

            centroids[count] = (cX, cY)
        
        return centroids
    
    def update(self, bounding_boxes, current_frame):
        """update the states of any objects"""
        template_corners = brisk_flannmatch(current_frame, template_image)

        # if target in frame
        if template_corners is not None:
            # reset the counter
            self.target_disappeared = 0  
            
        else:
            # if the target isn't in frame
            self.target_disappeared += 1
            if self.target_disappeared > self.maxDisappeared:
                self.TargetID = None 

        # if there has been nothing detected
        if len(bounding_boxes) == 0:
            for objectID in list(self.disappeared.keys()):
                # increase the number of frames they are disappeared for and remove if over the limit
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.unregister(objectID)

            return self.Objects, template_corners
        
        inputCentroids = tracker.calculate_centroids(bounding_boxes)

        # if there are no objects being tracked right now, just add everything to the list
        if len(self.Objects) == 0:
            for i in range(0, len(bounding_boxes)):
                self.register(bounding_boxes[i])

        # otherwise try do matching
        else:
            # get the keys and centroids of all the currently stored objects
            objectIds = list(self.Objects.keys())
            obj_centroids = tracker.calculate_centroids(self.Objects.values())

            # find the distances between each centroid and every existing one
            D = dist.cdist(np.array(obj_centroids), inputCentroids)

            # find the min value in each row, return indexes to sort in ascending order
            rows = D.min(axis = 1).argsort()

            # find the min value in each column, then sort them according to rows
            cols = D.argmin(axis=1)[rows]

            # variables to keep track of which rows and cols we have checked
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                # if they have already been checked, move on and ignore
                if row in usedRows or col in usedCols:
                    continue
                
                # get the object ID from the current row, set the new bounding box and reset disappeared counter
                objectID = objectIds[row]
                self.Objects[objectID] = bounding_boxes[col]
                self.disappeared[objectID] = 0

                # add it to the list to skip
                usedRows.add(row)
                usedCols.add(col)
            
            # now check for the sets that have not been matched with any existing object
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # if the number of tracked objects is greater than objects detected, check if they have disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over unused rows
                for row in unusedRows:
                    # get the object ID and increment the number of frames its disappeared
                    objectID = objectIds[row]
                    self.disappeared[objectID] += 1

                    # deregister if disappeared for too long
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.unregister(objectID)
            
            # if the number of objects detected is larger than tracked, register them
            else:
                for col in unusedCols:
                    self.register(bounding_boxes[col]) 

            # if the target has not been found yet and template has been found
            if self.TargetID == None and template_corners is not None:
                for objectID, bounding_box in self.Objects.items():
                    # first crop the input image to only look at the object detected
                    x, y, w, h = bounding_box

                    # check if all 4 corners are in the current bounding box
                    all_inside = all(x <= corner[0] <= x+w and y <= corner[1] <= y+h for corner in template_corners[:, 0])
                    
                    if all_inside:
                        self.TargetID = objectID
                        break

        return self.Objects, template_corners     