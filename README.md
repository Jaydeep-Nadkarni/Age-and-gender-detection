# Face and Age Detection

This project demonstrates a **Face Detection** and **Age Prediction** system using **OpenCV** and pre-trained **Caffe** models. The application captures video input from the webcam, detects faces, and predicts the age and gender of the person.

## Features 
    
- **Face Detection**: Detects faces in real-time using a pre-trained Caffe model.
- **Age Prediction**: Estimates the age group of a detected face based on a pre-trained age detection model.
- **Gender Prediction**: Identifies the gender (Male/Female) based on facial features.
- **Real-time Detection**: Works with your webcam to detect and predict age and gender.
- **Confidence Threshold**: Only faces with a confidence score greater than a threshold are processed.
- **Responsive Interface**: Displays detected information (Age and Gender) in real-time on the webcam feed. 

## Requirements

To run this project, you need the following: 

- **Python 3.x**
- **OpenCV**: For computer vision tasks like face detection and image processing.
- **NumPy**: For array manipulations.
- **Dlib** (optional for advanced facial features and emotion detection).

### Install Dependencies

You can install the required libraries using **pip**:

### Models
This project uses pre-trained models for face detection, age prediction, and gender prediction. You will need to download the following models:

Face Detection: res10_300x300_ssd_iter_140000_fp16.caffemodel and deploy.prototxt
Age Detection: age_net.caffemodel and age_deploy.prototxt
Gender Detection: gender_net.caffemodel and gender_deploy.prototxt
You can download the models from the following links:

Face Detection Model
Model: res10_300x300_ssd_iter_140000_fp16.caffemodel
Prototxt: deploy.prototxt
Author: OpenCV (Based on Caffe's SSD model)
Source: OpenCV Model Repository
Age Detection Model
Model: age_net.caffemodel
Prototxt: age_deploy.prototxt
Author: OpenCV (Pre-trained on a large dataset of faces)
Source: OpenCV Model Repository
Gender Detection Model
Model: gender_net.caffemodel
Prototxt: gender_deploy.prototxt
Author: OpenCV (Pre-trained on a large dataset of faces)
Source: OpenCV Model Repository
Place these models in a models/ directory within the project.

Usage
Clone this repository to your local machine:

```bash
Copy code
git clone https://github.com/Jaydeep-Nadkarni/Face-and-Age-Detection.git
cd Face-and-Age-Detection
Download the pre-trained models and place them in the models/ folder.
```

Run the application:

```bash
Copy code
python face_and_age_detection.py
The webcam feed will open, and faces detected in the video will have bounding boxes drawn around them. The system will also display the predicted age group and gender of the person in real-time.
```


### Key Press Actions:
Press 'q' to close the webcam and exit the application.
Code Explanation
Face Detection:
The Face Detection part uses OpenCV's DNN module with a pre-trained Caffe model (res10_300x300_ssd_iter_140000_fp16.caffemodel). This model identifies faces in real-time video frames captured by the webcam.

### Age and Gender Prediction:
The Age Detection Model (age_net.caffemodel) uses a pre-trained network that predicts the age group based on the detected face.
The Gender Detection Model (gender_net.caffemodel) uses a similar pre-trained model to predict the gender (Male or Female).
The age and gender predictions are displayed on the screen with the corresponding face detection bounding box.

### Real-Time Webcam Feed:
The system captures real-time frames from the webcam and processes each frame to detect faces, predict the age, and identify gender.

### Example Output
After running the code, you will see the webcam feed with bounding boxes around detected faces. The system will also display the predicted age group and gender, like:

### Troubleshooting
If the webcam feed doesn't open, ensure that you have a working webcam and that the required permissions are granted to access it.
If there are issues with missing model files, ensure you've downloaded and placed them correctly in the models/ folder.
Contributing
Feel free to fork this repository, contribute improvements, or submit bug reports. Contributions are welcome!

### Steps to Contribute:
Fork the repository
Clone your forked repository
Create a new branch
Make your changes and commit them
Push to your forked repository
Create a pull request with a clear description of the changes
License
This project is licensed under the MIT License - see the LICENSE file for details.

### Author
Jaydeep Nadkarni

GitHub: @Jaydeep-Nadkarni

Email: jaydeepnadkarni123@gmail.com

