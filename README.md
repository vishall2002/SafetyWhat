# SafetyWhat

1. Requirements
 -> Python 3.x
 -> OpenCV
 -> Ultralytics YOLO
 -> JSON
 -> pathlib

2. You can install the required libraries using pip
 -> pip install opencv-python ultralytics

3. Clone this repository to your local machine
4. Download the YOLOv8 model weights (yolov8n.pt) and keep in your project directory.
5. Place your input video file (e.g., vid13.mp4) in the project directory. If you want to use another vodeo replace it with you video name in the source file named as 'task2.py'
6. Open the script file (e.g., object_detection.py) in your preferred code editor (VS CODE)
7. Modify the input_video variable in the detect_objects_in_video function if your video file has a different name.
8. Run the script.
9. The script will process the video, display the detections in real-time, and save the detected objects and subobjects in the saved_objects and saved_subobjects folders, respectively.
10. That's all....you can now check in the json file, and the filder and sub-folders for dedected images.
