import numpy as np
import cv2
from scipy.spatial import distance as dist 
from collections import OrderedDict

template_path = "images\cat_image_2.jpg"

class tracker:
    def __init__(self, template_path:str, maxDisappeared = 50):
        """Takes in the maximum number of frames to be considered disappeared"""
        
        self.Objects = OrderedDict()    # stores the bounding boxes of each object
        self.disappeared = OrderedDict()# stores how many frames each object has disappeared
        self.nextObjectID = 0

        # keeps the tracking of the target
        self.target_disappeared = 0

        # key of the target
        self.TargetID = None

        self.maxDisappeared = maxDisappeared

        self.template = cv2.imread(template_path)

        # orb feature extractor
        self.brisk = cv2.BRISK_create()

        # create the flann based matcher
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6, # 12
                        key_size = 12,     # 20
                        multi_probe_level = 1) #2
        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.extract_template_features()

    # get the keypoints and features for the template image
    def extract_template_features(self):
        self.template_kp, self.template_des = self.brisk.detectAndCompute(self.template, None)

    # target to tell whether or if the object is the target
    def find_target(self, current_frame):
        transformed_corners = None
        object_kp, object_des = self.brisk.detectAndCompute(current_frame, None)

        if object_des is not None and len(object_des) > 2:
            matches = self.flann.knnMatch(self.template_des, object_des, k=2)
            matches = [match for match in matches if len(match) == 2]

            good_matches = []

            for m,n in matches:
                if m.distance < 0.67*n.distance:
                    good_matches.append(m)
            print(len(good_matches))

            if len(good_matches) > 10:
                template_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                current_frame_pts = np.float32([object_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                error_threshold = 5
                H, _ = cv2.findHomography(template_pts, current_frame_pts, cv2.RANSAC, error_threshold)

                if H is not None:
                    # Get the corners of the template in the larger image
                    h, w = self.template.shape[:2]
                    template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    transformed_corners = cv2.perspectiveTransform(template_corners, H)
                else:
                    print("H is none")

        return transformed_corners

    def get_target_ID(self):
        return self.TargetID
    
    def get_target_bbox(self):
        if self.TargetID is not None:
            return self.Objects[self.TargetID]
        else:
            return None

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
        template_corners = self.find_target(current_frame)

        # if target in frame
        if template_corners is not None:
            # reset the counter
            self.target_disappeared = 0  
            
        else:
            # if the target isn't in frame
            self.target_disappeared += 1
            if self.target_disappeared > self.maxDisappeared * 2:
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