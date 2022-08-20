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

PEOPLE_ID  = 14
VEHICLE_ID = 18

class Dataset:
    dir = "../cifar-100-python/"
    # Superclasses: 1) people 2) vehicle1
    def __init__(self):
        if not os.path.exists(self.dir):
            raise Exception("CIFAR-100 dataset is not downloaded")

    def load_training_data(self):
        train_dict = self.unpickle(self.dir + "train")

        data = train_dict['data']
        target = np.array(train_dict['coarse_labels'])

        # Select People and Vehicle1 superclass
        people_indexes = np.argwhere(target == PEOPLE_ID)
        vehicle1_indexes = np.argwhere(target == VEHICLE_ID)
        # Reshape from 2D to 1D np array for data retrieval and concatenate
        people_indexes = people_indexes.reshape((len(people_indexes),))
        vehicle1_indexes = vehicle1_indexes.reshape((len(vehicle1_indexes),))
        data_indexes = np.concatenate((people_indexes, vehicle1_indexes))

        train_data = data[data_indexes]
        train_target = target[data_indexes]

        return train_data, train_target        

    def load_testing_data(self):
        test_dict = self.unpickle(self.dir + "test")

        data = test_dict['data']
        target = np.array(test_dict['coarse_labels'])
        
        # Select People and Vehicle1 superclass
        people_indexes = np.argwhere(target == PEOPLE_ID)
        vehicle1_indexes = np.argwhere(target == VEHICLE_ID)
        # Reshape from 2D to 1D np array for data retrieval and concatenate
        people_indexes = people_indexes.reshape((len(people_indexes),))
        vehicle1_indexes = vehicle1_indexes.reshape((len(vehicle1_indexes),))
        data_indexes = np.concatenate((people_indexes, vehicle1_indexes))

        test_data = data[data_indexes]
        test_target = target[data_indexes]

        return test_data, test_target

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='latin1')
        return dict

def main(args: argparse.Namespace):
    dataset = Dataset()

    # Training stage
    if not args.test: 
        # Get the training data from Dataset
        train_data, train_target = dataset.load_training_data()

        # Define the classifier model
        mlp = MLPClassifier(max_iter=100)
        
        # Define the parameter space for MLP classifier
        # to find the best parameter set
        parameter_space = { 
            'hidden_layer_sizes': [(50,50,50), (100,), (20,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate': ['constant','adaptive'],
        }

        # Define GridSearch to find the best hyperparameters and their values
        # runs the exhaustive search over specified parameters
        # To speed up the process, we use all the processors to run the search (n_jobs=-1)
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
        clf.fit(train_data, train_target)

        # See the best parameters 
        print('Best parameters found:\n', clf.best_params_)

        # All results
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(clf, model_file)
    # Testing stage
    else:
        #Get the testing data from Dataset
        test_data, test_target = dataset.load_testing_data()

        # Load the model by deserializing
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Prediction based on the model
        predictions = model.predict(test_data)
        # Get classification probabilities
        pred_probabilities = model.predict_proba(test_data)[:, 1]

        # Evaluating the classification model
        evaluate(predictions, pred_probabilities, test_target)
    return

def evaluate(predictions, pred_probabilities, test_target):
    # Compute accurace
    print(accuracy_score(test_target, predictions))

    # Compute the confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(test_target, predictions)
    print("Confusion Matrix: ")
    print(confusion_matrix)

    # Compute macro-averaged precision and recall values
    precision, recall, _, _ = precision_recall_fscore_support(test_target, predictions, average='macro')
    print("Precision: ", precision)
    print("Recall", recall)

    # Plotting the results into a precision-recall curve space
    precision, recall, thresholds = precision_recall_curve(test_target, pred_probabilities, pos_label=18)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='purple')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true", help="Run on test data")
    parser.add_argument("--model_path", default="MLP_task1.model", type=str, help="Model path")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)