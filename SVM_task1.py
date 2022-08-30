import lzma
import pickle
import argparse
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score

# local 
from Dataset import Dataset

def main(args: argparse.Namespace):
    dataset = Dataset(subclasses=False)

    if not args.test:
        train_data, train_target = dataset.load_training_data()

        svc = SVC( kernel='rbf', probability=True)
        svc.fit(train_data, train_target)
        
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(svc, model_file)
    else:
        test_data, test_target = dataset.load_testing_data()

         # Load trained model
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)
        
        # Prediction based on the model
        predictions = model.predict(test_data)
        # Get classification probabilities
        pred_probabilities = model.predict_proba(test_data)

        # Evaluating the classification model
        evaluate(predictions, pred_probabilities, test_target)


def evaluate(predictions, pred_probabilities, test_target):
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

    print(test_target.shape)
    # Plotting the results into a precision-recall curve space
    precision, recall, _ = precision_recall_curve(test_target, pred_probabilities[:, 1], pos_label=18)
    
    _, ax = plt.subplots(1, 1, figsize=(6, 7), subplot_kw={'aspect': 'auto'})
    ax.plot(recall, precision, color='purple')
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=False, action="store_true", help="Run on test data")
    parser.add_argument("--model_path", default="SVM_task1.model", type=str, help="Model path")

    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)