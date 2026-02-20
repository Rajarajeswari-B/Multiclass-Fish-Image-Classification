🐟 Multiclass Fish Image Classification

📌 Project Overview
This project focuses on the automated classification of fish species using state-of-the-art Deep Learning techniques. By leveraging both custom Convolutional Neural Networks (CNNs) and Transfer Learning with various pre-trained architectures, the system can accurately identify fish categories from images. The project culminates in a user-friendly web application deployed via Streamlit for real-time inference.

🚀 Key Features
Data Augmentation : Robust preprocessing including rescaling, rotation, zooming, and flipping to prevent overfitting.
Hybrid Modeling : Comparison between a custom-built CNN and five industry-standard pre-trained models.
Comprehensive Evaluation : Detailed analysis using Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
Interactive Deployment : A Streamlit dashboard that allows users to upload images and receive instant classification results with confidence scores.

🛠️ Skills & Technologies
Languages : Python
Libraries : TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn
Computer Vision : OpenCV, PIL
Deployment : Streamlit
Deep Learning Concepts : CNN, Transfer Learning, Fine-tuning, Data Augmentation, Model Evaluation.

📂 Dataset
The project utilizes a multiclass fish dataset (e.g., "A Large-Scale Fish Dataset" from Kaggle).
Categories: Includes various species such as Gilt Head Bream, Red Sea Bream, Sea Bass, Red Mullet, etc.
Preprocessing: Images rescaled to [0,1] and resized to match model input requirements (e.g.,224×224).

🧠 Model Architecture & Approach
1. Data Preprocessing
Image Rescaling.
Augmentation (Rotation, Horizontal Flip, Zoom Range) to increase dataset diversity.
Split into Training, Validation, and Testing sets.

2. Training Strategy
CNN from Scratch : A custom sequential model to establish a baseline.
Transfer Learning : Leveraging weights from ImageNet using:
VGG16
ResNet50
MobileNet
InceptionV3
EfficientNetB0
Fine-tuning : Adjusting the top layers of pre-trained models to suit the fish classification task.

3. Evaluation
Comparison of training/validation loss and accuracy curves.
Evaluation on the test set using a Classification Report.
Selection of the Max Accuracy Model for deployment.

📊 Business Use Cases
Marine Biodiversity Monitoring : Automating the identification of fish species for researchers and environmentalists.
Fisheries Management: Assisting in sorting and cataloging fish in commercial settings.
Educational Tools: Helping students and hobbyists identify fish species instantly.

🖥️ Application Preview
Upload : Users can upload .jpg or .png images.
Prediction : The app displays the most likely fish species.
Confidence : A progress bar shows the model's confidence level.

🤝 Contributing
Contributions are welcome! Please fork the repository and create a pull request with your suggested changes.

Developed by Raji
Connect with me on LinkedIn - https://www.linkedin.com/in/rajarajeswari-baladhandapani/
