# Spam-E-Mail-Classification using Neural Network 

This project demonstrates a simple email spam detection system using neural networks implemented in Python with Keras. The model is trained on a dataset of emails labeled as spam or non-spam (ham).

# Dataset
The dataset used for training the model is stored in a CSV file named emails.csv. It contains two columns: "Email Text" and "Prediction" where "Prediction" indicates whether an email is spam (1) or not spam (0).

# Dependencies
* Python 3.x

* numpy

* matplotlib

* pandas

* keras

* scikit-learn

* imbalanced-learn (for SMOTE oversampling)

* Install dependencies using pip:


pip install numpy matplotlib pandas keras scikit-learn imbalanced-learn

# Usage
1) Clone the repository or download the code files.
2) Place the emails.csv dataset file in the same directory as the code files.
3) Run the spam_detection.py script.

python spam_detection.py
Input an email text when prompted.

Enter the email text: This is an example email.
The program will output whether the email is classified as spam or not.

# Model Architecture

The neural network model architecture consists of several dense layers with ReLU activation functions and dropout regularization. The final layer uses a sigmoid activation function for binary classification.

# Training
The model is trained using the Adam optimizer with mean squared error loss. The dataset is split into training and testing sets, and SMOTE oversampling is applied to handle class imbalance.

# Results
The trained model's performance is evaluated on the testing set using accuracy as the metric.

# Authors
Kanishk Maheshwari
