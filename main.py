from video_process_unit import VideoProcessor

MainProcessor = VideoProcessor(save_data = False)
# Processor.analyze_video()
# Processor.analyze_video('data/video/video1.mp4')
MainProcessor.analyze_img('data/photos/photo1.jpg')
# Processor.analyze_img()