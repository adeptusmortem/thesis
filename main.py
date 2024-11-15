from video_process_unit import VideoProcessor

MainProcessor = VideoProcessor(save_data = False, distance_estimator_type = "c", model_path = "yolo11m-pose.pt", focal_length = 800)
# MainProcessor.analyze_video()
# MainProcessor.analyze_video('data/video/video1.mp4')
# MainProcessor.analyze_img('data/photos/photo3.jpg')
MainProcessor.analyze_img()