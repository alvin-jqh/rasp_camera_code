import cv2
import numpy as np
import mediapipe as mp

import time

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()
row_size = 50  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 255)  # red
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

# Label box parameters
label_text_color = (255, 255, 255)  # white
label_font_size = 1
label_thickness = 2
        
class object_detector:
    def __init__(self, model:str, max_results:int = 5, score_threshold: float = 0.6):
        """
        Args:
            model: Name of the TFLite object detection model.
            max_results: Max number of detection results.
            score_threshold: The score threshold of detection results.
        """
        self.model_path = model
        self.max_results = max_results
        self.score_threshold = score_threshold
        self.detection_result_list = []
        self.detection_frame = None

        self.intialise_detector()
    
    def save_result(self, result: vision.ObjectDetectorResult, 
                    unused_output_image: mp.Image, timestamp_ms: int):
        
        global FPS, COUNTER, START_TIME

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        self.detection_result_list.append(result)
        COUNTER += 1


    def intialise_detector(self):
        # initialises the model options
        self.base_options = python.BaseOptions(model_asset_path = self.model_path)
        self.options = vision.ObjectDetectorOptions(base_options = self.base_options,
                                                    running_mode = vision.RunningMode.LIVE_STREAM,
                                                    max_results = self.max_results,
                                                    score_threshold = self.score_threshold,
                                                    category_allowlist = ["person"],
                                                    result_callback = self.save_result)
        
        # intialise
        self.detector = vision.ObjectDetector.create_from_options(self.options)

    def convert_cv2mp(self, image:np.uint8):
        """ converts an opencv image to a mediapipe image, returns mp_image"""

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        return mp_image
    
    def run_detector(self, mp_image):
        # do some detecting 
        self.detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    def visualize(self, image, detection_result) -> np.ndarray:
        """Draws bounding boxes on the input image and return it.
        Args:
            image: The input RGB image.
            detection_result: The list of all "Detection" entities to be visualized.
        Returns:
            Image with bounding boxes.
        """

        MARGIN = 10  # pixels
        ROW_SIZE = 30  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (0, 0, 0)  # black

    # array containing all the results, scores, and bounding boxes
        bounding_boxes = []

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            # Use the orange color for high visibility.
            cv2.rectangle(image, start_point, end_point, (0, 165, 255), 3)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                            MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            
            bounding_boxes.append((bbox.origin_x, bbox.origin_y, bbox.width, bbox.height))

        return image, bounding_boxes

    def loop_function(self, image:np.uint8):
        """run this function in while loop, 
        returns the results in form (start_point, end_point)"""
        detection_results = []

        # step 1 convert the image into a mediapipe image
        mp_image = self.convert_cv2mp(image)

        # step 2 run the detector
        self.run_detector(mp_image)

        # step 3 visualise results

        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image.copy()
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if self.detection_result_list:
            current_frame, detection_results = self.visualize(current_frame, self.detection_result_list[0])

            self.detection_frame = current_frame

            self.detection_result_list.clear()

        return detection_results  

    def get_annotated_image(self):
        return self.detection_frame    
    
    def end_loop(self):
        """ close the detector"""
        self.detector.close()

class image_segmentor:
    def __init__(self, model:str):
        """Continuously run inference on images acquired from the camera.

        Args:
            model: Name of the TFLite object detection model.
        """
        self.model_path = model
        self.mask_results = None
        self.RGB_mask = None
        self.binary_mask = None
        self.masked_feed = None

        self.initialise_segmentor()

    def save_result(self, result, 
                    unused_output_image: mp.Image, timestamp_ms: int):
        self.mask_results = result

    def initialise_segmentor(self):
        """Gets all the options and initialises the image segmentor"""

        self.base_options = python.BaseOptions(model_asset_path = self.model_path)
        self.options = vision.ImageSegmenterOptions(base_options = self.base_options,
                                                    running_mode = vision.RunningMode.LIVE_STREAM,
                                                    output_category_mask = True,
                                                    result_callback = self.save_result)
        
        # intialise
        self.segmentor = vision.ImageSegmenter.create_from_options(self.options)

    def convert_cv2mp(self, image:np.uint8):
        """ converts an opencv image to a mediapipe image, returns mp_image"""

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        return mp_image
    
    def run_detector(self, mp_image):
        """ do some detecting """
        self.segmentor.segment_async(mp_image, time.time_ns() // 1_000_000)

    def create_mask(self, category_mask):
        """Creates an RGB and Binary mask, returns both"""

        MASK_COLOR = np.array([0, 0, 0], dtype=np.uint8) # black
        BG_COLOR = np.array([255, 255, 255], dtype=np.uint8) # white

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        condition_b = category_mask.numpy_view() > 0.2

        fg_image = np.zeros(condition.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(condition.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        RGB_mask = np.where(condition, fg_image, bg_image)
        binary_mask = np.where(condition_b, np.zeros(condition_b.shape[:2],dtype=np.uint8), np.ones(condition_b.shape[:2], dtype=np.uint8))


        return RGB_mask, binary_mask
    
    def loop_function(self, image:np.uint8, type:bool = False):
        """run this function in while loop
        Returns:
            Binary or RGB mask"""

        # step 1 convert to mp image
        mp_image = self.convert_cv2mp(image)

        # step 2 run the model
        self.run_detector(mp_image)

        # step 3 create mask
        if self.mask_results:
            self.RGB_mask, self.binary_mask = self.create_mask(self.mask_results.category_mask)
            
            self.mask_results = None
        if type:
            return self.RGB_mask
        else:
            return self.binary_mask

    def get_mask(self, type:bool = False):
        """returns one of the masks, by default returns binary
            Args:
            type: True returns the RGB mask, False returns the binary mask"""
        if type:
            return self.RGB_mask
        else:
            return self.binary_mask
        
    def get_masked_feed(self, image):
        if self.binary_mask is not None:
            self.masked_feed = image * self.binary_mask
        else:
            print("Binary mask is none")

    def end_loop(self):
        self.segmentor.close()

class gesture_recogniser:
    def __init__(self, model: str, num_hands: int, min_hand_detection_confidence: float,
        min_hand_presence_confidence: float, min_tracking_confidence: float):
        """Continuously run inference on images acquired from the camera.

            Args:
                model: Name of the gesture recognition model bundle.
                num_hands: Max number of hands can be detected by the recognizer.
                min_hand_detection_confidence: The minimum confidence score for hand
                    detection to be considered successful.
                min_hand_presence_confidence: The minimum confidence score of hand
                    presence score in the hand landmark detection.
                min_tracking_confidence: The minimum confidence score for the hand
                    tracking to be considered successful.
            """

        self.model_path = model
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_hand_presence_confidence = min_hand_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.recognition_result_list = []

        self.initialise_recogniser()


    def save_result(self, result:vision.GestureRecognizerResult, unused_output_image: mp.Image, timestamp_ms: int):
        self.recognition_result_list.append(result)

    def initialise_recogniser(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        canned_gesture_classifier_options = python.components.processors.ClassifierOptions(category_allowlist = ["None", "Thumb_Up", "Thumb_Down"])
        options = vision.GestureRecognizerOptions(base_options=base_options,
                                                running_mode=vision.RunningMode.LIVE_STREAM,
                                                num_hands=self.num_hands,
                                                min_hand_detection_confidence=self.min_hand_detection_confidence,
                                                min_hand_presence_confidence=self.min_hand_presence_confidence,
                                                min_tracking_confidence=self.min_tracking_confidence,
                                                canned_gesture_classifier_options=canned_gesture_classifier_options,
                                                result_callback=self.save_result)
        
        self.recogniser = vision.GestureRecognizer.create_from_options(options)

    def convert_cv2mp(self, image:np.uint8):
        """ converts an opencv image to a mediapipe image, returns mp_image"""

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        return mp_image
    
    def run_recogniser(self, mp_image):
        # do some detecting 
        self.recogniser.recognize_async(mp_image, time.time_ns() // 1_000_000)

    def draw_hands(self, original_current_frame:np.uint8, recognition_result_list):
        gesture_detected = None
        current_frame = original_current_frame.copy()
        recognition_frame = None
        for hand_index, hand_landmarks in enumerate(recognition_result_list[0].hand_landmarks):
            # Calculate the bounding box of the hand
            x_min = min([landmark.x for landmark in hand_landmarks])
            y_min = min([landmark.y for landmark in hand_landmarks])
            y_max = max([landmark.y for landmark in hand_landmarks])

            # Convert normalized coordinates to pixel values
            frame_height, frame_width = current_frame.shape[:2]
            x_min_px = int(x_min * frame_width)
            y_min_px = int(y_min * frame_height)
            y_max_px = int(y_max * frame_height)

            # Get gesture classification results
            if recognition_result_list[0].gestures:
                gesture = recognition_result_list[0].gestures[hand_index]
                category_name = gesture[0].category_name
                gesture_detected = category_name
                score = round(gesture[0].score, 2)
                result_text = f'{category_name} ({score})'

                # Compute text size
                text_size = \
                cv2.getTextSize(result_text, cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                                label_thickness)[0]
                text_width, text_height = text_size

                # Calculate text position (above the hand)
                text_x = x_min_px
                text_y = y_min_px - 10  # Adjust this value as needed

                # Make sure the text is within the frame boundaries
                if text_y < 0:
                    text_y = y_max_px + text_height

                # Draw the text
                cv2.putText(current_frame, result_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_DUPLEX, label_font_size,
                            label_text_color, label_thickness, cv2.LINE_AA)

            # Draw hand landmarks on the frame
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                            z=landmark.z) for landmark in
            hand_landmarks
            ])
            mp_drawing.draw_landmarks(
            current_frame,
            hand_landmarks_proto,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        recognition_frame = current_frame

        return recognition_frame, gesture_detected
    

    def loop_function(self, image:np.uint8):
        """"run this function inside the while loop,
            returns the anotated image and gesture seen"""
        
        gestures = []
        recognition_frame = None

        # step 1 convert the image into a mediapipe image
        mp_image = self.convert_cv2mp(image)

        # step 2 run the recogniser
        self.run_recogniser(mp_image)
        current_frame = image.copy()

        # step 3 draw the hands
        if self.recognition_result_list:
            
            recognition_frame, gestures = self.draw_hands(current_frame, self.recognition_result_list)
            self.recognition_result_list.clear()

        return recognition_frame, gestures
        
    def end_loop(self):
        self.recogniser.close() 

import sys
if __name__ == "__main__":
    model_path = "gesture_recognizer.task"

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    gr = gesture_recogniser(model=model_path, num_hands=1, min_hand_detection_confidence=0.5,
                            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            ) 

        image = cv2.flip(image,1)

        recognition_frame, gesture = gr.loop_function(image) 
        print(gesture)

        cv2.imshow("fram", image)

        if recognition_frame is not None:
            cv2.imshow("recog", recognition_frame)

        if cv2.waitKey(1) == 27:
            break   

    gr.end_loop
    cap.release()
    cv2.destroyAllWindows()