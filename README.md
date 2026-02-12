Implements a CNN-based image classification system for distinguishing between cats and dogs. The workflow follows a standard deep learning pipeline using TensorFlow/Keras, making it suitable for beginners learning CNNs.

First, the required libraries are imported, including NumPy, Matplotlib, and Keras components such as Conv2D, MaxPooling2D, Flatten, Dense, and Dropout. These are essential for building and training a convolutional neural network.

The dataset is loaded from CSV files (input.csv, labels.csv, and their test equivalents). Instead of loading images directly, pixel values are stored numerically in CSV format. The input data is reshaped into a 4D tensor structure suitable for CNNs:
(number of samples, image height, image width, color channels). This step is crucial because CNNs expect image-like input.

The CNN model is built using a Sequential architecture. It starts with convolution layers (Conv2D) to extract spatial features such as edges and patterns, followed by MaxPooling layers to reduce dimensionality and computation while preserving important features. The Flatten layer converts 2D feature maps into a 1D vector so it can be processed by fully connected layers.

A Dense layer with a sigmoid activation function is used in the output layer, indicating a binary classification problem. The model is compiled using the Adam optimizer and binary cross-entropy loss, which is appropriate for two-class classification.

During prediction, the model outputs a probability value. A threshold of 0.5 is applied to decide the class label. Based on this result, the model prints whether the image is predicted as a cat or dog.

Finally, the trained model is saved as model.h5, allowing reuse without retraining. Overall, this notebook clearly demonstrates how CNNs can be applied to image classification in a simple and structured manner, making it ideal for academic learning and exam preparation.
