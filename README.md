# Live_Emoji
Welcome to the Live Emoji project! This project demonstrates how to classify different human emotions using a deep learning model built with TensorFlow and Keras. The system leverages real-time webcam input to predict and display emotions in live video feed, using MediaPipe's Holistic model for face and hand landmark detection.

Key objectives of this project include:

Loading and processing facial and hand landmark data.
Building and training a neural network to classify emotions.
Performing real-time inference through webcam input.
Displaying predicted emotions on a live video stream.

The project is divided into two parts:

a. Data Loading and Training: Involves loading landmark data, training the neural network model, and saving the trained model for future use.

b. Inference: Once the model is trained, real-time emotion classification is performed on live video input via a webcam.

Requirements

To run this project, the following dependencies are required:

- Python 3.7 or higher
- TensorFlow (2.x)
- Keras
- OpenCV
- MediaPipe
- NumPy

Instructions to Run the Project

1. Clone the Repository:
Download the project repository to your local machine.

2. Data Loading and Model Training:
Ensure that the required .npy files (containing facial and hand landmark data) are in the project directory. Execute the training script to load the data, build and train the neural network, and save the trained model and labels for inference.

3. Running Inference on Live Video:
After the model has been trained, run the inference script to start the webcam. The system will detect facial and hand landmarks in real time and predict the associated emotion, displaying it on the video stream.

4. Modify the Code:
The project code is modular and flexible. You can modify the neural network architecture in the training script or fine-tune the inference logic in the inference script as needed.

Project Structure

The project contains:

- A script for loading and training the model
- A script for performing real-time emotion prediction
- The trained model and label files

How the Project Works

1. Data Loading:
Facial and hand landmark data are loaded from .npy files. The data is concatenated and shuffled to create a training dataset.

2. Model Training:
A neural network with two hidden layers is trained on the processed landmark data, and the model predicts one of several emotion categories using softmax activation.

3. Inference:
The trained model is used to classify emotions in real time, utilizing face and hand landmarks detected from live video input.

Future Work

Some potential improvements for this project include:

- Expanding the dataset to include more emotion categories.
- Experimenting with different neural network architectures to improve accuracy.
- Adding more advanced data augmentation techniques.

Conclusion

Live Emoji is an engaging project that showcases the power of neural networks in real-time video applications. With the help of MediaPipe for landmark detection and TensorFlow for model training and inference, the system dynamically responds to human emotions, making it an excellent demonstration of AI in interactive systems.

Feel free to explore, modify, and enhance the code. If you have any feedback or suggestions, contributions are welcome!
