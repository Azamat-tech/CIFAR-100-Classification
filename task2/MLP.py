from asyncio import constants
import os
import lzma
import argparse
import pickle
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

START_INDEX = 70
END_INDEX   = 75

class Dataset:
    dir = "../cifar-100-python/"
    # Fine classes of Superclass People
    def __init__(self):
        if not os.path.exists(self.dir):
            raise Exception("CIFAR-100 dataset is not downloaded")

    def load_training_data(self):
        train_dict = self.unpickle(self.dir + "train")

        data = train_dict['data']
        target = np.array(train_dict['fine_labels'])

        # Select fine classes of People superclass
        fine_indexes = []
        for i in range(START_INDEX, END_INDEX):
            fine_index = np.argwhere(target == i)
            fine_index = fine_index.reshape((len(fine_index),))
            fine_indexes.append(fine_index)

        fine_indexes = np.concatenate(fine_indexes)
        
        # Get data of those fine indexes
        train_data = data[fine_indexes]
        train_target = target[fine_indexes]

        return train_data, train_target

    def load_testing_data(self):
        test_dict = self.unpickle(self.dir + "test")

        data = test_dict['data']
        target = np.array(test_dict['fine_labels'])

        # Select fine classes of People superclass
        fine_indexes = []
        for i in range(START_INDEX, END_INDEX):
            fine_index = np.argwhere(target == i)
            fine_index = fine_index.reshape((len(fine_index),))
            fine_indexes.append(fine_index)

        fine_indexes = np.concatenate(fine_indexes)
        
        # Get data of those fine indexes
        test_data = data[fine_indexes]
        test_target = target[fine_indexes]

        return test_data, test_target

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

def main(args: argparse.Namespace):
    dataset = Dataset()

    # Training stage
    if not args.test: 
        train_data, train_target = dataset.load_training_data()

        # Scaling the data to be between 0 and 1
        sc = StandardScaler()
        train_data = sc.fit_transform(train_data)

        # Reduce dimensiality of subclasses' pixels into 2 components
        # We want the amount of variance explained to be 95%
        pca = PCA(n_components=0.95)
        train_data = pca.fit_transform(train_data)

        # Serialize the PCA transformation.
        with lzma.open("pca", "wb") as transform:
            pickle.dump(pca, transform)

        model = MLPClassifier(
            hidden_layer_sizes=(50,50,50), 
            activation='relu', 
            solver='adam',
            alpha=0.001,
            learning_rate='constant'
        ).fit(train_data, train_target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    # Testing stage
    else:
        #Get the testing data from Dataset
        test_data, test_target = dataset.load_testing_data()
        
        # Scaling the data to be between 0 and 1
        sc = StandardScaler()
        test_data = sc.fit_transform(test_data)

        # Load the PCA and reduce dimensiality of testing data
        with lzma.open("pca", "rb") as transform:
            pca = pickle.load(transform)
            # Reduce dimensiality of testing data as well
            test_data = pca.transform(test_data)

        # Load trained model
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Prediction based on the model
        predictions = model.predict(test_data)

        # Get classification probabilities
        pred_probabilities = model.predict_proba(test_data)

        # Evaluating the classification model
        evaluate(predictions, pred_probabilities, test_target, model.classes_)
    return

def evaluate(predictions, pred_probabilities, test_target, classes):
    # Compute accuracy
    print("Accuracy Score: ", accuracy_score(test_target, predictions))

    # Compute the confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(test_target, predictions)
    print("Confusion Matrix: ")
    print(confusion_matrix)

    # Compute macro-averaged precision and recall values
    p, r, _, _ = precision_recall_fscore_support(test_target, predictions, average='macro')
    print("Precision: ", p)
    print("Recall", r)

    # Plotting the results into a precision-recall curve space
    precision, recall, _ = precision_recall_curve(test_target, pred_probabilities[:, 4], pos_label=classes[4])
    # precision, recall, threshold = sklearn.metrics.precision_recall_curve(test_target, prediction_probs[:, 4], pos_label=24)
    
    _, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true", help="Run on test data")
    parser.add_argument("--model_path", default="MLP_task2.model", type=str, help="Model path")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
