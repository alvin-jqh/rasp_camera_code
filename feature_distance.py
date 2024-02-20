import cv2
import numpy as np

class match_distance:
    def __init__(self):
        # orb feature extractor
        self.orb = cv2.ORB_create()

        # create the flann based matcher
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6, # 12
                        key_size = 12,     # 20
                        multi_probe_level = 1) #2
        search_params = dict(checks=50)   # or pass empty dictionary

        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_keypoints_descriptors(self, image, bbox):
        x, y, w, h = bbox

        # Crop the image to the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Detect keypoints and compute descriptors for the cropped image
        kp, des = self.orb.detectAndCompute(cropped_image, None)

        for point in kp:
            point.pt = (point.pt[0] + x, point.pt[1] + y)

        return kp, des
    
    def get_good_matches_coordinates(self, matches, left_kp, right_kp):
        good_matches_coordinates = []

        for match in matches:
            left_idx = match.queryIdx
            right_idx = match.trainIdx

            # Get the coordinates of the keypoints in the left and right images
            left_point = left_kp[left_idx].pt
            right_point = right_kp[right_idx].pt

            # Append the coordinates to the list
            good_matches_coordinates.append((left_point, right_point))

        return good_matches_coordinates

    def left_right_match(self, left_image, right_image, left_bbox, right_bbox):
        matched_image = None
        good_matches_coordinates = None

        # Extract keypoints and descriptors within the bounding boxes
        left_kp, left_des = self.extract_keypoints_descriptors(left_image, left_bbox)
        right_kp, right_des = self.extract_keypoints_descriptors(right_image, right_bbox)

        if left_des is not None and len(left_des) > 2 and right_des is not None and len(right_des) > 2:
            matches = self.flann.knnMatch(left_des, right_des, k=2)
            matches = [match for match in matches if len(match) == 2]

             # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            good_matches = []

            # ratio test as per Lowe's paper
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.5*n.distance:
                    matchesMask[i]=[1,0]
                    good_matches.append(m)

            draw_params = dict(matchColor = (0,255,0),
                            singlePointColor = (255,0,0),
                            matchesMask = matchesMask,
                            flags = cv2.DrawMatchesFlags_DEFAULT)
            matched_image = cv2.drawMatchesKnn(left_image,left_kp,right_image,right_kp,matches,None,**draw_params)

             # Get the coordinates of good matches
            if len(good_matches) > 50:
                good_matches_coordinates = self.get_good_matches_coordinates(good_matches, left_kp, right_kp)

        return matched_image, good_matches_coordinates
    
    
    def find_distances(self, matched_coordinates, baseline, focal_length):
        distances = []
        x_coords = []
        for left_point, right_point in matched_coordinates:
            xL, _ = left_point
            xR, _ = right_point

            disparity = xL-xR

            if disparity != 0:
                distance = (baseline * focal_length) / disparity
                if distance < 500:
                    distances.append(abs(distance))
                    x_coords.append(xL)

        avg_distance = np.mean(distances)
        avg_x = int(np.mean(x_coords))
        return avg_distance, avg_x
