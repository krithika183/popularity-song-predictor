POPULARITY-SONG-PREDICTION

STEP 1.Load the Dataset: Reads Spotify_data.csv into a pandas DataFrame.

STEP 2. Data Cleaning: Drops any non-numeric columns from the DataFrame.

STEP 3.Target Variable Creation: Converts the Popularity column into a binary target (0 or 1) based on the threshold of 65.

STEP 4.Class Distribution Check: Prints the counts of each class (popular/not popular) before applying resampling.

STEP 5.SMOTE Application:

To address potential class imbalance, the Synthetic Minority Over-sampling Technique (SMOTE) is applied.

SMOTE is only executed if the minority class has at least 6 samples; otherwise, a warning is printed, and the original imbalanced data is used.

STEP 6.Train-Test Split: The dataset (after potential SMOTE application) is split into training (80%) and testing (20%) sets. A random_state of 42 ensures reproducibility.

STEP 7.Feature Scaling:

StandardScaler is used to standardize the features (mean=0, variance=1) of both the training and testing sets. This is crucial for optimal neural network performance.

STEP 8.Class Weight Calculation:

compute_class_weight is used to calculate weights that penalize misclassifications of the minority class more heavily. These weights are passed to the model during training to combat imbalance.

STEP 9.ANN Model Construction:

A Sequential Keras model is built with:

An input layer (Dense) with 64 neurons, relu activation.

A Dropout layer (0.3) to prevent overfitting.

A hidden layer (Dense) with 32 neurons, relu activation.

Another Dropout layer (0.3).

An output layer (Dense) with 1 neuron and sigmoid activation, suitable for binary classification.

The model is compiled using binary_crossentropy as the loss function, adam as the optimizer, and Precision, Recall, and accuracy as evaluation metrics.

STEP 10.Model Training: The ANN is trained for 100 epochs with a batch_size of 16. A validation_split of 0.2 is used, and the pre-calculated class_weights_dict is applied during training.

STEP 11.Model Evaluation:

Predictions are made on the scaled test set.

The raw probabilities are converted into binary predictions (0 or 1) using a 0.5 threshold.

The test accuracy and a comprehensive classification report (detailing precision, recall, f1-score, and support for each class) are printed.

STEP 12.Training History Plot: A plot is generated to visualize the model's accuracy and val_accuracy over the training epochs, which helps in understanding the learning process and detecting overfitting
