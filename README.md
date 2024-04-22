## Shop Name Detection and Extraction
This project aims to detect shop names from a video source, extract and save the shop names along with their corresponding start and end timestamps.

## Features
* **Data Collection:** Videos and frames were collected as the primary data source for training the object detection model
* **Annotation:** Annotations were performed using CVAT (Computer Vision Annotation Tool) to label shop names in the video frames.
* **Training:** Utilized Ultralytics YOLOv8 for training the object detection model on the annotated data.
* **Optical Character Recognition (OCR):** Uses PaddleOCR to extract text from the detected regions.
* **Video Processing:** Handles video input and output using OpenCV.
* **Data Saving:** Saves the detected shop names and their timestamps in a CSV file.


## Prerequisites
* Python 3.x
* OpenCV
* Ultralytics YOLO
* PaddleOCR
* argparse
* datetime
* csv

## Installation
1. Clone the repository:
     > git clone https://github.com/govardhanbilla5/shop-name-detection.git
2. Install the required packages:
     > pip install -r requirements.txt
## Usage
Run the **'main.py'** script with the required arguments:\
  $python main.py --intput_source <path_to_video_file> --weights <path_to_model_weights>
### Arguments
* --intput_source: Path to the video source file.
* --weights: Path to the YOLO model weights file. Default is best.pt.
## Output
* CSV file: Contains the detected shop names along with their start and end timestamps.
* The CSV file will be saved in the project directory with a timestamped filename.
## License
This project is licensed under the MIT License. See LICENSE for more details.
## Acknowledgments
Ultralytics YOLO for object detection\
PaddleOCR for optical character recognition\
OpenCV for video processing
