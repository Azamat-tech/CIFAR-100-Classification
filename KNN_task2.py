import lzma
import pickle
import argparse
import sklearn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# local 
from Dataset import Dataset

def main(args: argparse.Namespace):
    # Load Subclass datasets
    dataset = Dataset(subclasses=True)

    # Training
    if not args.test:
        train_data, train_target = dataset.load_training_data()

        # Standardize the training data
        sc = StandardScaler()
        train_data = sc.fit_transform(train_data)

        # Instead of selecting the # components manually,
        # We want the amount of variance explained to be 95%
        pca = PCA(n_components=0.95)
        train_data = pca.fit_transform(train_data)

        # Serialize the PCA transformation.
        with lzma.open(args.pca_path, "wb") as transform:
            pickle.dump(pca, transform)

        best_k = get_best_k(train_data, train_target)
        model = KNeighborsClassifier(n_neighbors=best_k).fit(train_data, train_target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
    # Testing
    else:
        test_data, test_target = dataset.load_testing_data()

        # Scaling the testing data
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



def get_best_k(train_data, train_target):
    """
        This method find the best value for K in KNN classfier
        using the cross_val_score and returns the best parameter
    """
    num_folds = 10
    # Prepare the K paramters
    k_range = list(range(1, 31))
    # Cross validation to choose k from 1 to 31.
    k_scores = []
    for i in k_range:
        model = KNeighborsClassifier(n_neighbors=i, weights="distance")
        cv_scores = cross_val_score(model, train_data, train_target, cv=num_folds, scoring="accuracy")
        k_scores.append(np.mean(cv_scores))

    # Choose hyperparameter with lowest mean cross validation error
    return np.argmax(k_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true", help="Run on test data")
    parser.add_argument("--model_path", default="KNN_task2.model", type=str, help="Model path")
    parser.add_argument("--pca_path", default="pca_KNNtask2", type=str, help="PCA path")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
