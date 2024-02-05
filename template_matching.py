import cv2
import numpy as np
import matplotlib.pyplot as plt

def template_match(input_image, template):
    
    # Initialize feature detectors
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(input_image, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    # Use a matcher (e.g., Brute Force) to find matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.65 * n.distance:
            good_matches.append(m)

    # Check if enough good matches are found
    if len(good_matches) > 10:
        return True
    else:
        return False
    
def sift_bfmatch(input_image, template):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both the large image and the template
    keypoints_large, descriptors_large = sift.detectAndCompute(input_image, None)
    keypoints_template, descriptors_template = sift.detectAndCompute(template, None)

    if descriptors_large is not None and descriptors_template is not None:
        # Create a Brute Force Matcher object
        bf = cv2.BFMatcher()

        # Match descriptors of the template and the larger image
        matches = bf.knnMatch(descriptors_template, descriptors_large, k=2)

        # Apply ratio test to get good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                good_matches.append(m)
    
    transformed_corners = None

    if len(good_matches) > 15:
        # Get coordinates of key points in both the template and the larger image
        template_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        large_image_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        H, _ = cv2.findHomography(template_pts, large_image_pts, cv2.RANSAC)

        # makes sure the ransac algorithm doesnt return None and crash the program
        if H is not None:
            # Get the corners of the template in the larger image
            h, w = template.shape[:2]
            template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(template_corners, H)

    return transformed_corners

def orb_flannmatch(left_image, right_image):
    # Initialize SIFT detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for both the large image and the template
    keypoints_left, descriptors_left = orb.detectAndCompute(left_image, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(right_image, None)

    matches = None
    transformed_corners = None
    matched_coordinates = []

    if descriptors_left is not None and descriptors_right is not None:
        #FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6, 
                        key_size = 12,     
                        multi_probe_level = 1) 
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descriptors_right,descriptors_left,k=2)
        # print(matches)

        good_matches = []
        matches = [match for match in matches if len(match) == 2]
        matches = sorted(matches, key=lambda x: x[0].distance)

        #lowes ratio test, find paper
        for m,n in matches:
            if m.distance < 0.65*n.distance:
                good_matches.append(m)
        
        print(len(good_matches))
    
        if len(good_matches) > 10:
            for match in good_matches:
                left_idx = match.queryIdx
                right_idx = match.trainIdx

                left_point = tuple(int(coord) for coord in keypoints_left[left_idx].pt)
                right_point = tuple(int(coord) for coord in keypoints_right[right_idx].pt)

                matched_coordinates.append((left_point, right_point))

    return matched_coordinates

def orb_bfmatch(input_image, template):
    # Initialize SIFT detector
    orb = cv2.ORB_create()

    # Detect keypoints and compute descriptors for both the large image and the template
    keypoints_large, descriptors_large = orb.detectAndCompute(input_image, None)
    keypoints_template, descriptors_template = orb.detectAndCompute(template, None)

    matches = []

    if descriptors_large is not None and descriptors_template is not None:
        # Create a Brute Force Matcher object
        bf = cv2.BFMatcher()

        # Match descriptors of the template and the larger image
        matches = bf.knnMatch(descriptors_template, descriptors_large, k = 2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        print(len(good_matches))
    
    transformed_corners = None

    if len(good_matches) > 10:
        print("detected")
        # Get coordinates of key points in both the template and the larger image
        template_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        large_image_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        error_threshold = 5
        H, _ = cv2.findHomography(template_pts, large_image_pts, cv2.RANSAC, error_threshold)

        # makes sure the ransac algorithm doesnt return None and crash the program
        if H is not None:
            # Get the corners of the template in the larger image
            h, w = template.shape[:2]
            template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(template_corners, H)
        else:
            print("H is none")

    return transformed_corners

def draw_orb_bf(input_image, template):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(template,None)
    kp2, des2 = orb.detectAndCompute(input_image,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(template,kp1,input_image,kp2,matches[:30],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def draw_sift_bf(img2, img1):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3),plt.show()

def draw_siftflann(img2, img1):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()

def draw_orbflann(img2, img1):
    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                    table_number = 6, # 12
                    key_size = 12,     # 20
                    multi_probe_level = 1) #2
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    
    matches = flann.knnMatch(des1,des2,k=2)
    matches = [match for match in matches if len(match) == 2]
    # # Need to draw only good matches, so create a mask
    # matchesMask = [[0,0] for i in range(len(matches))]
    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.75*n.distance:
    #         matchesMask[i]=[1,0]       
    # draw_params = dict(matchColor = (0,255,0),
    #                 singlePointColor = (255,0,0),
    #                 matchesMask = matchesMask,
    #                 flags = cv2.DrawMatchesFlags_DEFAULT)
    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    # cv2.imshow("Matches", img3)

    goodMatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodMatches.append(m)
    print(len(goodMatches))
    if len(goodMatches) > 10:
        # reshape(-1,1,2) -> reshapes to (nKeypoints, 1, 2)
        srcPts = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2) 
        dstPts = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        errorThreshold = 5
        M, mask = cv2.findHomography(srcPts,dstPts,cv2.RANSAC,errorThreshold)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape[:2]
        imgBorder = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        warpedImgBorder = cv2.perspectiveTransform(imgBorder,M)
        img2 = cv2.polylines(img2,[np.int32(warpedImgBorder)],True,(255, 0 ,0),3, cv2.LINE_AA)
    else:
        print("Not enough matches")
        matchesMask = None

    green = (0,255,0)
    drawParams = dict(matchColor=green,singlePointColor=None,matchesMask=matchesMask,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,goodMatches,None,**drawParams)
    
    cv2.imshow("draw", img3)

def brisk_flannmatch(input_image, template):
    # Initialize SIFT detector
    brisk = cv2.BRISK_create()

    # Detect keypoints and compute descriptors for both the large image and the template
    keypoints_large, descriptors_large = brisk.detectAndCompute(input_image, None)
    keypoints_template, descriptors_template = brisk.detectAndCompute(template, None)

    matches = None
    transformed_corners = None

    if descriptors_large is not None and descriptors_template is not None:
        #FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6, 
                        key_size = 12,     
                        multi_probe_level = 1) 
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(descriptors_template,descriptors_large,k=2)
        # print(matches)

        good_matches = []
        matches = [match for match in matches if len(match) == 2]

        #lowes ratio test, find paper
        for m,n in matches:
            if m.distance < 0.67*n.distance:
                good_matches.append(m)
        
        # print(len(good_matches))
    

        if len(good_matches) > 10:
            print("detected")
            # Get coordinates of key points in both the template and the larger image
            template_pts = np.float32([keypoints_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            large_image_pts = np.float32([keypoints_large[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find the homography matrix
            error_threshold = 5
            H, _ = cv2.findHomography(template_pts, large_image_pts, cv2.RANSAC, error_threshold)

            # makes sure the ransac algorithm doesnt return None and crash the program
            if H is not None:
                # Get the corners of the template in the larger image
                h, w = template.shape[:2]
                template_corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                transformed_corners = cv2.perspectiveTransform(template_corners, H)
            else:
                print("H is none")

    return transformed_corners

def brisk_flannmatch_coords(left_image, right_image):
    # Initialize BRISK detector
    brisk = cv2.BRISK_create()

    # Detect keypoints and compute descriptors for both the large image and the template
    keypoints_left, descriptors_left = brisk.detectAndCompute(left_image, None)
    keypoints_right, descriptors_right = brisk.detectAndCompute(right_image, None)

    matches = None
    transformed_corners = None
    matched_coordinates = []

    if descriptors_left is not None and descriptors_right is not None:
        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)   # or pass an empty dictionary

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_right, descriptors_left, k=2)
        # print(matches)

        good_matches = []
        matches = [match for match in matches if len(match) == 2]
        matches = sorted(matches, key=lambda x: x[0].distance)

        # Lowe's ratio test, find paper
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            for match in good_matches:
                left_idx = match.queryIdx
                right_idx = match.trainIdx

                # Check if the keypoints have valid 'pt' attribute
                if 0 <= left_idx < len(keypoints_left) and 0 <= right_idx < len(keypoints_right):
                    left_point = tuple(int(coord) for coord in keypoints_left[left_idx].pt)
                    right_point = tuple(int(coord) for coord in keypoints_right[right_idx].pt)

                    matched_coordinates.append((left_point, right_point))

    return matched_coordinates
    
if __name__ == "__main__":

    input_image = cv2.imread("R1.png")
    template_image = cv2.imread("L1.png")

    # draw_orb_bf(input_image, template_image)
    # draw_siftflann(input_image, template_image)
    draw_orbflann(input_image.copy(), template_image.copy())
    pairs = orb_flannmatch(input_image.copy(), template_image.copy())

    for i in range(20):
        left_point, right_point = pairs[i]

        cv2.circle(template_image,right_point, 4, (50 + 7*i, 100, 12 * i), -1)
        cv2.circle(input_image,left_point, 4, (50 + 7*i, 100, 12 * i), -1)
    
    left_point, right_point = pairs[1]
    cv2.circle(template_image,right_point, 4, (0, 0, 255), -1)
    cv2.circle(input_image,left_point, 4, (0, 0, 255), -1)

    # box = brisk_flannmatch(input_image, template_image)
    # image_with_box = input_image.copy()
    # if box is not None:
    #     cv2.polylines(image_with_box, [np.int32(box)], True, (0, 255, 0), 2)

    cv2.imshow("template", template_image)
    cv2.imshow("input", input_image)
    # cv2.imshow("box", image_with_box)


    # if template_match(input_image, template_image):
    #     print("success")

    # else:
    #     print("fucked up")

    if cv2.waitKey(0) == 27:
        cv2.destroyAllWindows()