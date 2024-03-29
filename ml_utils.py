from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def prepare_mnist_data():
    # Load MNIST data from sklearn
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist["data"], mnist["target"]
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train.values, X_test.values, y_train.values, y_test.values

def plots_data(X, y, sample):
    # Select a random sample to plot
    indices = np.random.choice(range(len(X)), sample, replace=False)
    X_sample = X[indices]
    y_sample = y[indices]
    
    fig, axes = plt.subplots(1, sample, figsize=(12, 3))
    for ax, image, label in zip(axes, X_sample, y_sample):
        ax.set_axis_off()
        ax.imshow(image.reshape(28, 28), cmap='gray')
        ax.set_title('Label: %s' % label)
    plt.show()

def ml_train(X, y, model_type='logistic_regression'):
    """
    Train a model on the MNIST dataset.
    
    Parameters:
    - X: Features for training.
    - y: Labels for training.
    - model_type: Type of model to train. Options are 'logistic_regression', 'svm', 'random_forest'.
    
    Returns:
    - Trained model.
    """
    if model_type == 'logistic_regression':
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_type == 'svm':
        model = SVC(random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type. Choose from 'logistic_regression', 'svm', 'random_forest'.")
    
    model.fit(X, y)
    return model

def ml_predict(x, ml_model):
    # Make predictions using the model
    return ml_model.predict([x])[0]

def plots_predict(X, y_true, ml_model, sample):
    # Select a random sample to make predictions
    indices = np.random.choice(range(len(X)), sample, replace=False)
    X_sample = X[indices]
    y_sample_true = y_true[indices]
    
    fig, axes = plt.subplots(1, sample, figsize=(12, 3))
    for ax, image, true_label in zip(axes, X_sample, y_sample_true):
        pred_label = ml_predict(image, ml_model)
        ax.set_axis_off()
        ax.imshow(image.reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color='green' if true_label == pred_label else 'red')
    plt.show()

def accuracy_score_ml(x_test, y_test, ml_model):
    # Estimate the accuracy of the model
    predictions = ml_model.predict(x_test)
    return accuracy_score(y_test, predictions)


if __name__ == '__main__':
    # Example usage:
    X_train, X_test, y_train, y_test = prepare_mnist_data()
    plots_data(X_train, y_train, 10)
    model = ml_train(X_train, y_train, 'random_forest')
    print("Accuracy on test set:", accuracy_score_ml(X_test, y_test, model))
    plots_predict(X_test, y_test, model, 10)
