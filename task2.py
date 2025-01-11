# Importing essential libraries
import cv2
from ultralytics import YOLO
import json
from pathlib import Path

# Loading the YOLO model
model = YOLO('yolov8n.pt')

# Input and output paths of various folders, sub-folders, etc.
def detect_objects_in_video():
    input_video = 'vid13.mp4'  
    obj_folder = 'saved_objects'  
    subobj_folder = 'saved_subobjects'  
    
    # Creating folders if they do not exist
    Path(obj_folder).mkdir(parents=True, exist_ok=True)
    Path(subobj_folder).mkdir(parents=True, exist_ok=True)
    
    # Checking whether the video file exists
    if not Path(input_video).exists():
        print(f"Error: {input_video} not found!")
        return
    
    # Opening video file
    video = cv2.VideoCapture(input_video)
    if not video.isOpened():
        print("Error: Unable to open the video file!")
        return
    
    obj_id = 0  # To count objects
    all_results = []  # Store all detections for JSON output
    
    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break  
            
            # Run YOLO model on the current frame and iterating loop throuth detected objects
            detections = model(frame)
            for box in detections[0].boxes:
                obj_id += 1
                
                # Get bounding box coordinates and class name
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                label = detections[0].names[int(box.cls)]
                if label not in ['person', 'cat']:      # Process specefc obejcts
                    continue
                
                # Crop the detected object
                cropped_object = frame[y1:y2, x1:x2]
                obj_file = f"{obj_folder}/{label}_{obj_id}.jpg"
                cv2.imwrite(obj_file, cropped_object)
                
                # Create detection dictionary
                obj_data = {
                    "object": label,
                    "id": obj_id,
                    "bbox": [x1, y1, x2, y2],
                    "subobject": None  
                }
                
                # Check for subobjects within the detected object
                if cropped_object.size > 0:
                    sub_detections = model(cropped_object)
                    for sub_box in sub_detections[0].boxes:
                        sx1, sy1, sx2, sy2 = map(int, sub_box.xyxy[0].tolist())
                        sub_label = sub_detections[0].names[int(sub_box.cls)]
                        
                        if sub_label not in ['helmet', 'doors', 'cars']:    # Process specefic subobject
                            continue
                        
                        # Crop the subobject
                        cropped_subobject = cropped_object[sy1:sy2, sx1:sx2]
                        if cropped_subobject.size > 0:
                            sub_file = f"{subobj_folder}/{sub_label}_{obj_id}.jpg"
                            cv2.imwrite(sub_file, cropped_subobject)
                        
                        # Add subobject details
                        obj_data["subobject"] = {
                            "object": sub_label,
                            "id": obj_id,
                            "bbox": [sx1 + x1, sy1 + y1, sx2 + x1, sy2 + y1]
                        }
                        break  
                
                # Add object data to results
                all_results.append(obj_data)
                
                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {obj_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the video with detections
            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  
                break
    
    except Exception as e:
        print(f"Error occurred: {e}")

# Saving results to the json file and releasing all resources 
    finally:
        with open('detection_results.json', 'w') as json_file:
            json.dump(all_results, json_file, indent=4)
        
        video.release()
        cv2.destroyAllWindows()

# Main function 
if __name__ == "__main__":
    detect_objects_in_video()
