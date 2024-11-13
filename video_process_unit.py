import cv2
from yolo_process_unit import HumanDetector
from data_process_unit import DataProcessor
from datetime import datetime
import csv
import os

class VideoProcessor():
    def __init__(self, confidence_threshold = 0.3, focal_lengh = 400, save_data = False, path='data/saves/'):
        """ Init function. """
        self.Detector = HumanDetector(confidence_threshold=confidence_threshold, focal_lengh=focal_lengh) # Creating HumanDetector object
        self.Processor = DataProcessor() # Creating data processor
        self.save_data = save_data
        self.path = path

    def analyze_video(self, path=0):
        """ Analyze video stream. """
        # Use 0 for the default camera, else path to video
        # cap = cv2.VideoCapture("data/video/video1.mp4")  
        cap = cv2.VideoCapture(path)
        # Define frame size
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, detections = self.Detector.frame_processor(frame)

            # Show the processed image
            cv2.imshow("HumanDetector, q to close window", frame)
            
            # Save data if allowed
            self.data_saver(frame, detections)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Exit on 'q'
                break
            elif key == ord('s'):  # Save on 's'
                # Create a timestamped filename
                filename = datetime.now().strftime("data/saves/screenshot_%Y%m%d_%H%M%S.png")
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
        cap.release()
        cv2.destroyAllWindows()

    def analyze_img(self, path="test.png"):
        """ Analyze single image. """
        # Load and preprocess the image
        image_path = path
        frame = cv2.imread(image_path)
        frame, detections = self.Detector.frame_processor(frame)

        # Save data if allowed
        self.data_saver(frame, detections)

        cv2.imshow("HumanDetector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Save on 's'
            # Create a timestamped filename
            filename = datetime.now().strftime("data/saves/screenshot_%Y%m%d_%H%M%S.png")
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def data_saver(self, frame, detections):
        if self.save_data:
            export_data = self.Processor.export_data(detections)
            if export_data:
                # filename =  datetime.now().strftime(self.path+"img_%Y%m%d_%H%M%S.png")
                filename =  self.path + f"{export_data[0]['unix_timestamp']}.png"
                file_path = self.path + "export_data.csv"
                cv2.imwrite(filename, frame)

                # Check if the file already exists
                file_exists = os.path.exists(file_path)

                # Open the file in append mode if it exists; write mode if it does not
                with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames = export_data[0].keys(), delimiter=';')
                    
                    # If the file doesn't exist, write the header
                    if not file_exists:
                        writer.writeheader()
                    
                    # Write the rows to the CSV file
                    writer.writerows(export_data)



