# Drone Landing Detection Project

## Project Overview

This project focuses on detecting whether a drone is properly aligned over a landing pad during flight using video analysis and machine learning techniques.

### Key Components:
- **Video Processing:** Capturing and analyzing video frames of drone flights.
- **Landing Pad Detection:** Using YOLOv8 to detect the borders of the landing pad in each frame.
- **Feature Extraction:** Extracting coordinates and area of the landing pad from detected borders.
- **Classification:** Training an SVM model to classify if the drone is centered ("Aligned") or not ("Misaligned") based on the extracted features.

## Repository Structure

- All code related to the project is located in the `scripts` branch.
- The `main` branch contains general project files and documentation.
- The project uses multiple Python scripts, each responsible for different steps:
  - Parsing the input MP4 video into individual JPG frames.
  - Extracting detection data from video frames.
  - Generating CSV files with labeled data.
  - Training and evaluating the SVM classifier.
  - Running real-time predictions on video input.

## How It Works

1. **Video Parsing:**  
   A script parses the input MP4 video into individual JPG frames for analysis.

2. **YOLOv8 Detection:**  
   A custom-trained YOLOv8 model detects the landing pad's borders in video frames.

3. **Data Extraction:**  
   Coordinates of the landing pad corners and the bounding box area are extracted from the detection results.

4. **CSV Generation:**  
   This data is saved into a CSV file (`obb_data.csv`), which is then used for model training.

5. **SVM Training:**  
   An SVM classifier is trained on the extracted features to distinguish aligned vs misaligned drone positions.

6. **Video Prediction:**  
   The trained SVM model is applied on live video frames to classify alignment in real-time, with results annotated on the video.

## Usage

- Clone the repository.
- Switch to the `scripts` branch to access all Python scripts.
- Follow the README instructions in each script for running video parsing, detection, training, and prediction.

---

For questions or contributions, please contact [your email or GitHub profile].
