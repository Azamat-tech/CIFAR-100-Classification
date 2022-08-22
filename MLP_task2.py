import lzma
import argparse
import pickle
import sklearn
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# local
from Dataset import Dataset

def main(args: argparse.Namespace):
    dataset = Dataset(subclasses=True)

    # Training stage
    if not args.test: 
        train_data, train_target = dataset.load_training_data()

        # Scaling the data to be between 0 and 1
        sc = StandardScaler()
        train_data = sc.fit_transform(train_data)

        # Instead of selecting the # components manually,
        # We want the amount of variance explained to be 95%
        pca = PCA(n_components=0.95)
        train_data = pca.fit_transform(train_data)

        # Serialize the PCA transformation.
        with lzma.open(args.pca_path, "wb") as transform:
            pickle.dump(pca, transform)

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
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=10)
        clf.fit(train_data, train_target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(clf, model_file)

    # Testing stage
    else:
        #Get the testing data from Dataset
        test_data, test_target = dataset.load_testing_data()
        
        # Scaling the data to be between 0 and 1
        sc = StandardScaler()
        test_data = sc.fit_transform(test_data)

        # Load the PCA and reduce dimensiality of testing data
        with lzma.open(args.pca_path, "rb") as transform:
            pca = pickle.load(transform)
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
    
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'auto'})
    ax.plot(recall, precision, color='purple')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true", help="Run on test data")
    parser.add_argument("--model_path", default="MLP_task2.model", type=str, help="Model path")
    parser.add_argument("--pca_path", default="pca_MLPtask2", type=str, help="PCA path")
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
