import cv2
import numpy as np
import logging
import os
import zipfile
from flask import Flask, request, jsonify
import requests

# Suppress logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Paths to your model files
vehicle_orientation_cfg = 'yolov4_vehicle_orientation.cfg'
vehicle_orientation_weights = 'yolov4_vehicle_orientation_74500.weights'
vehicle_orientation_names = 'vehicle_orientation.names'

# Load YOLOv4 model for front-facing vehicle detection
net = cv2.dnn.readNetFromDarknet(vehicle_orientation_cfg, vehicle_orientation_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class names for vehicle orientation model
with open(vehicle_orientation_names, 'r') as f:
    vehicle_orientation_classes = [line.strip() for line in f.readlines()]

# Define front-facing vehicle classes
front_facing_classes = ['car_front', 'bus_front', 'truck_front', 'motorcycle_front', 'cycle_front']

# Roboflow API details
ROBOFLOW_API_KEY = "bQ066VC2s60TyMB7V384"
ROBOFLOW_API_URL = "https://detect.roboflow.com/ambulance-hkj51/1"

# Initialize Flask app
app = Flask(__name__)

def process_video(file_path):
    cap = cv2.VideoCapture(file_path)
    results = []
    snapshot_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (500, 500))
        blob = cv2.dnn.blobFromImage(resized_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

        detections = net.forward(output_layers)
        filtered_detections = []

        for output in detections:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, width, height = (detection[0:4] * np.array([500, 500, 500, 500])).astype('int')
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    class_name = vehicle_orientation_classes[class_id]
                    if class_name in front_facing_classes:
                        if x >= 0 and y >= 0 and x + width <= 500 and y + height <= 500:
                            filtered_detections.append((x, y, int(width), int(height), class_name, confidence))

        indices = cv2.dnn.NMSBoxes(
            [d[:4] for d in filtered_detections],
            [d[5] for d in filtered_detections],
            score_threshold=0.5,
            nms_threshold=0.4
        )
        filtered_detections = [filtered_detections[i] for i in indices.flatten()]

        for (x, y, w, h, class_name, confidence) in filtered_detections:
            crop_img = resized_frame[y:y+h, x:x+w]
            if crop_img.size > 0:
                _, crop_img_encoded = cv2.imencode('.jpg', crop_img)
                response = requests.post(
                    ROBOFLOW_API_URL,
                    files={"file": crop_img_encoded.tobytes()},
                    headers={"Authorization": f"Bearer {ROBOFLOW_API_KEY}"}
                )
                result = response.json()
                for prediction in result.get('predictions', []):
                    if prediction['class'] == 'Ambulance':
                        results.append({
                            'class': 'Ambulance',
                            'confidence': confidence,
                            'box': [x, y, w, h]
                        })
                    else:
                        results.append({
                            'class': class_name,
                            'confidence': confidence,
                            'box': [x, y, w, h]
                        })

        # Take snapshot every 10 seconds
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % (fps * 10) == 0:
            snapshot_path = f'snapshot_{snapshot_count}.jpg'
            cv2.imwrite(snapshot_path, resized_frame)
            results.append({'snapshot': snapshot_path})
            snapshot_count += 1

    cap.release()
    return results

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['videos']
    zip_path = "input_videos.zip"
    file.save(zip_path)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("videos")

    results = {}
    for video_file in os.listdir("videos"):
        video_path = os.path.join("videos", video_file)
        video_results = process_video(video_path)
        results[video_file] = video_results
        os.remove(video_path)

    os.remove(zip_path)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
