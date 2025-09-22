# FaceIdentify Face Recognition System

## Project Overview
This project implements a face recognition system based on Backpropagation (BP) Neural Network and Support Vector Machine (SVM), using the CASIA-FaceV5 dataset for training and testing. The system provides a complete machine learning workflow, including data preprocessing, model training, performance evaluation, and result visualization, aiming to offer a simple yet effective face recognition solution.

## Features
- Supports two classic classification algorithms: BP Neural Network and SVM
- Complete dataset automatic partitioning functionality
- Precise face image preprocessing
- Detailed training process logging and loss monitoring
- Multi-dimensional model performance evaluation
- Algorithm performance comparison analysis
- Intuitive training result visualization display
- Supports Chinese display in charts

## Technical Architecture

### Core Algorithms
- **BP Neural Network**: A three-layer feedforward neural network including input layer, hidden layer, and output layer, using sigmoid activation function and softmax output layer
- **Support Vector Machine (SVM)**: Using RBF kernel function, supporting probability output

### Image Processing Flow
- Image reading and RGB conversion
- Size normalization (32×32)
- Pixel value standardization (0-1)
- Feature flattening

### Model Evaluation Metrics
- Accuracy
- Loss function value
- Training time
- Confusion matrix

## Installation Requirements

### Environment Requirements
- Python 3.6+
- Operating System: Windows/Linux/MacOS

### Dependencies
```
numpy==1.20.3
matplotlib==3.4.3
scikit-learn==1.0.2
pillow==8.3.2
```

### Installation Method
1. Clone or download the project to your local machine
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
FaceIdentify/
├── 64_CASIA-FaceV5/      # Face dataset, containing 180 categories of face images
├── src/                  # Source code directory
│   ├── data/             # Data processing module
│   ├── plotting/         # Result visualization module
│   ├── training/         # Model training module
│   ├── testing/          # Model testing module
│   └── utils/            # Utility functions
├── labels/               # Label files directory
│   └── classes.json      # Class names file
├── results/              # Results storage directory
│   ├── comparison/       # Model comparison results
│   └── plots/            # Chart results
├── plots/                # Temporary charts and logs
├── main.py               # Main program entry
├── requirements.txt      # Project dependencies
└── README.md             # Project description document
```

## Usage Instructions

### Basic Usage
1. Ensure the dataset is correctly placed in the `64_CASIA-FaceV5/` directory
2. Run the main program
   ```bash
   python main.py
   ```
3. The program will automatically execute the following process:
   - Dataset partitioning
   - BP Neural Network training
   - BP Neural Network testing
   - SVM model training and testing
   - Model performance comparison
   - Result visualization

### Custom Configuration
You can adjust the model configuration by modifying parameters in the code:
- BP Neural Network parameters (hidden layer size, learning rate, number of training epochs, etc.)
- SVM parameters (C value, kernel function type, gamma parameter, etc.)
- Image preprocessing parameters (image size, etc.)

## Algorithm Description

### BP Neural Network
BP Neural Network is a multi-layer feedforward network trained according to error backpropagation algorithm. The network structure implemented in this project is as follows:
- Input layer: 32×32×3=3072 neurons (corresponding to preprocessed image features)
- Hidden layer: 128 neurons, using sigmoid activation function
- Output layer: 180 neurons (corresponding to 180 categories in the dataset), using softmax activation function

The training process uses mini-batch gradient descent, and the loss function uses cross-entropy loss.

### SVM Classifier
Support Vector Machine is a binary classification model that can be extended to multi-class problems through kernel tricks. In this project, we use:
- RBF kernel function
- C=1.0 regularization parameter
- gamma='scale' kernel coefficient setting

## Dataset Description
This project uses the CASIA-FaceV5 face dataset, which contains:
- 180 different face categories (one folder per person)
- Each category contains multiple face images with different poses and expressions
- Image format is .bmp

When running the program, the system will automatically use 1 image from each category as the test set, and the rest as the training set.

## Result Output

### Training Results
- Loss values during training are displayed in real-time on the console
- Trained models are saved as .pkl files

### Testing Results
- Test accuracy is displayed on the console
- Detailed test logs are saved in `plots/test_log.txt`

### Visualization Results
- Loss curves during training
- Accuracy comparison charts of different models
- Result images are saved in the `results/plots/` directory

### Comparison Results
- Numerical comparison data of different model performances
- Comparison results are saved in the `results/comparison/` directory

## Notes
1. Ensure the dataset path is correct and contains sufficient face images
2. The training process may take some time, depending on hardware performance
3. If you need to adjust model parameters to obtain better performance, please modify the corresponding training files
4. In case of Chinese display issues, the program has built-in font settings to ensure normal display of Chinese in charts
5. Result files are automatically saved in the specified directories, no manual creation required

## License
MIT License

## Acknowledgments
Thanks to the providers of the CASIA-FaceV5 dataset, as well as all researchers and developers who have contributed to the fields of machine learning and computer vision.