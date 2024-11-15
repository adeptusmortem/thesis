import cv2
from ultralytics import YOLO
from datetime import datetime
import numpy as np

class HumanDetector:
    def __init__(self, model_path='yolo11n-pose.pt', confidence_threshold = 0.1, focal_length = 400, distance_estimator_type = "rm"):
        """ Init function, adjusting params of detector. """
        self.model = YOLO(model_path) # Load yolo.pt model
        self.confidence_threshold = confidence_threshold # Set threshold for detection
        self.focal_length = focal_length # Focal lengh, camera parameter
        self.distance_estimator_type = distance_estimator_type # Type of distance estimator
        """
        rm - rotation matrix
        ad - angle difference rotation
        s - simple
        c - combo
        """
        self.average_shoulder_width = 0.40 # average human shoulder width is 40 cm
        self.average_shoulder_height = 1.7 # average human height is 170 cm
        self.origin_x = None # x of center of the image
        self.origin_y = None # y of center of the image
        
    def frame_processor(self, frame):
        """ Main process function. """
        self.origin_x = frame.shape[1] // 2 # x of center of the image
        self.origin_y = frame.shape[0] // 2 # y of center of the image
        detections = []

        results = self.model(frame) # Get result form model
        for result in results:
            keypoints = result.keypoints.xy  # Assumes YOLO-Pose model provides keypoints
            for index, box in enumerate(result.boxes):
                # Filter detections by confidence threshold
                if box.conf[0] < self.confidence_threshold:
                    continue

                # Calculate distance from pose keypoints
                keypoint = keypoints[index]
                match self.distance_estimator_type:
                    case "rm":
                        distance = self.distance_estimator(keypoint)
                    case "ad":
                        distance = self.distance_estimator_v2(keypoint)
                    case "s":
                        distance = self.distance_estimator_fast(box)
                    case "c":
                        x1 = np.array([int(keypoint[5][0]), int(keypoint[5][1])])
                        x2 = np.array([int(keypoint[6][0]), int(keypoint[6][1])])
                        if int(x1[0])*int(x1[1])*int(x2[0])*int(x2[1]) > 0:
                            v1 = x2 - x1
                            shoulder_base = np.linalg.norm(v1) # Shoulder base in pixels
                            x1, y1, x2, y2 = box.xyxy[0]
                            bbox_width = x2 - x1
                            k = float(shoulder_base / bbox_width) # Ratio of shoulder base to bbox width
                            d1 = self.distance_estimator(keypoint)
                            d2 = self.distance_estimator_v2(keypoint)
                            d3 = self.distance_estimator_fast(box)
                            distance = k*0.5*d1 + k*0.5*d2 + (1-k)*d3
                        # if d1*d2 > 0:                            
                        else:
                            distance = self.distance_estimator_fast(box)
                    case _:
                        distance = self.distance_estimator_fast(box)

                center_coordinates = self.center_estimator(keypoint)
                x, y = center_coordinates
                center_offset = x - self.origin_x, self.origin_y - y # offset from the center of a frame

                # Process frame (and shoulder line)
                        
                x1 = (int(keypoint[5][0]), int(keypoint[5][1]))
                x2 = (int(keypoint[6][0]), int(keypoint[6][1]))
                x3 = (int(keypoint[11][0]), int(keypoint[11][1]))
                x4 = (int(keypoint[12][0]), int(keypoint[12][1]))

                if x1[0] * x1[1] * x2[0] * x2[1] > 0:
                    cv2.line(frame, x1, x2, (255,0,0), 5)
                if x4[0] * x4[1] * x3[0] * x3[1] > 0:
                    cv2.line(frame, x2, x4, (255,0,0), 5)
                    cv2.line(frame, x4, x3, (255,0,0), 5)
                    cv2.line(frame, x3, x1, (255,0,0), 5)
                
                self.visualisator(frame, box, distance, center_coordinates)

                # Collect data into list
                
                # Current Unix timestamp in seconds
                unix_timestamp = datetime.now().timestamp()
                detection = {"unix_timestamp":unix_timestamp,
                             "distance":distance,
                             "center_offset":center_offset,
                             "frame_resolution":(self.origin_x, self.origin_y),
                             "probability":float(box.conf[0].numpy())}
                detections.append(detection)

        return frame, detections
    
    def distance_estimator(self, keypoint): 
        """ Measure distance from camera to middle point of shoulders. """
        """ 0: Nose 1: Left Eye 2: Right Eye 3: Left Ear 4: Right Ear ...
            5: Left Shoulder 6: Right Shoulder 7: Left Elbow 8: Right Elbow ...
            9: Left Wrist 10: Right Wrist 11: Left Hip 12: Right Hip ...
            13: Left Knee 14: Right Knee 15: Left Ankle 16: Right Ankle """
        
        x1 = np.array([int(keypoint[5][0]), int(keypoint[5][1]), 0])
        x2 = np.array([int(keypoint[6][0]), int(keypoint[6][1]), 0])
        x3 = np.array([int(keypoint[11][0]), int(keypoint[11][1]), 0])
        x4 = np.array([int(keypoint[12][0]), int(keypoint[12][1]), 0])

        x2vec = lambda y1, y2: y2[:2] - y1[:2]
                
        i = 0
        j = 0
        alpha = 0
        beta = 0
        rate = 0.3
        loop_threshold = 10
        while True:
            if i > loop_threshold:
                break
            i = i + 1

            while True:
                if j > loop_threshold:
                    break
                j = j + 1
                        
                v1 = x2vec(x1, x2)
                v2 = x2vec(x3, x4)
                cross_product = np.cross(v1, v2)
                is_close = np.allclose(cross_product, 0.0, rtol = 0.01)

                if is_close:
                    break
                
                step = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
                alpha = alpha + rate*step
                Rx = np.array([[1,0,0],
                            [0, np.cos(alpha), -np.sin(alpha)],
                            [0, np.sin(alpha), np.cos(alpha)]])
                x1 = np.matmul(Rx, x1)
                x2 = np.matmul(Rx, x2)
                x3 = np.matmul(Rx, x3)
                x4 = np.matmul(Rx, x4)


            step = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            beta = beta + rate*step

            Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                        [0, 1, 0],
                        [-np.sin(beta), 0, np.cos(beta)]])
            x1 = np.matmul(Ry, x1)
            x2 = np.matmul(Ry, x2)
            x3 = np.matmul(Ry, x3)
            x4 = np.matmul(Ry, x4)
        
        shoulder_base = np.linalg.norm(v1)
        distance = 1
        
        distance = self.focal_length / shoulder_base * self.average_shoulder_width
        if float(distance):
            return float(distance)      
        return 0.0

    def distance_estimator_v2(self, keypoint):
        """ Measure distance from camera to the midpoint of shoulders. """
        x1 = np.array([int(keypoint[5][0]), int(keypoint[5][1]), 0])
        x2 = np.array([int(keypoint[6][0]), int(keypoint[6][1]), 0])
        x3 = np.array([int(keypoint[11][0]), int(keypoint[11][1]), 0])
        x4 = np.array([int(keypoint[12][0]), int(keypoint[12][1]), 0])

        v1 = x2[:2] - x1[:2]
        v2 = x4[:2] - x3[:2]

        # Angle differance
        angle_diff = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))

        # Angle correction
        if angle_diff > 1e-2:
            correction_angle = angle_diff * 0.3
            R = np.array([[np.cos(correction_angle), -np.sin(correction_angle)],
                        [np.sin(correction_angle), np.cos(correction_angle)]])
            v1 = R.dot(v1)
        
        shoulder_base = np.linalg.norm(v1)
        distance = self.focal_length / shoulder_base * self.average_shoulder_width if shoulder_base else float('inf')

        if float(distance):
            return float(distance)
        return 0

    def distance_estimator_fast(self, box):
        x1, y1, x2, y2 = box.xyxy[0]
        
        # Height and width of bounding box in pixels
        bbox_height = y2 - y1  
        bbox_width = x2 - x1
        aspect_ratio = bbox_width / bbox_height
        
        # Use different size estimations based on aspect ratio (pose)
        if aspect_ratio < 0.75:  # likely standing
            distance = (self.average_shoulder_height * self.focal_length) / bbox_height
        elif aspect_ratio < 1.5:  # likely sitting
            distance = (self.average_shoulder_width * self.focal_length) / bbox_width
        else:  # likely lying down
            distance = (self.average_shoulder_width * self.focal_length) / bbox_width

        return float(distance)

    def center_estimator(self, keypoint):
        """ Calculate coordinates of middle point of sholders. """
        left_shoulder = int(keypoint[5][0]), int(keypoint[5][1])
        right_shoulder = int(keypoint[6][0]), int(keypoint[6][1])
        center_coordinates = int((left_shoulder[0] + right_shoulder[0]) / 2), int((left_shoulder[1] + right_shoulder[1]) / 2)
        return center_coordinates

    def visualisator(self, frame, box, distance, center_coordinates):
        """ Visualize processed data. """
        # Bounding box coordinates
        bbox_coords = box.xyxy[0]
        x1, y1, x2, y2 = bbox_coords

        # Calculate the center of the human
        center_x, center_y = center_coordinates
        if center_x < x1 and center_y < y1:
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)

        # Draw the bounding box and data on the frame
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"Distance: {distance:.2f} m", # , M({center_x - self.origin_x}, {self.origin_y - center_y})
                    (center_x + 10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Person {box.conf[0]:.2f}", (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw red cross at the middle of the image
        # cv2.line(frame, (self.origin_x, 0), (self.origin_x, self.origin_y*2), (0,0,255),1)
        # cv2.line(frame, (0, self.origin_y), (self.origin_x*2, self.origin_y), (0,0,255),1)
        # cv2.putText(frame, f"{self.origin_x, self.origin_y}", (self.origin_x, self.origin_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # resize = self.resizer(frame, height=920)
        return frame

    
    def resizer(self, frame, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = frame.shape[:2]

        if width is None and height is None:
            return frame
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(frame, dim, interpolation=inter)