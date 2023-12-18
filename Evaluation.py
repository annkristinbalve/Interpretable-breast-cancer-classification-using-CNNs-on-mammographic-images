import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
import seaborn as sns
import pandas as pd

## plot validation and training loss and accuracy
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    # plot loss over epochs
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plot accuracy over epochs
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


## plot confusion matrix
def plot_confusion_matrix(y_true, y_pred_classes):
    # Compute confusion matrix
    # Define class labels
    class_names = ['Normal', 'Benign', 'Malignant']
    cm = confusion_matrix(y_true, y_pred_classes)

    # Create plot and plot confusion matrix with true classes on the y-axis and predicted labels on the x-axis
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', square=True, cbar=True, xticklabels=class_names, yticklabels=class_names,
               annot_kws={"fontsize":12})
    plt.xlabel('Predicted classes', fontsize = 12)
    plt.ylabel('True classes', fontsize = 12)
    plt.title('Confusion matrix', fontsize = 12)
    plt.tight_layout()
    plt.show()

# compute specificity
def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm[0, 0] / (cm[0, 0] + cm[0, 1])

# plot roc curve
def plot_roc_curve(model, x_test, y_test):
    """
    Plots the ROC curve for a multi-class classification model.
    """
    y_pred = model.predict(x_test)
    n_classes = y_test.shape[1]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # store false positives and true positives for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.figure(figsize=(8, 6))
    # plot the roc curve for each class
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for class %d' % (roc_auc[i], i))
    
    ## add labels
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize = 12)
    plt.ylabel('True Positive Rate', fontsize = 12)
    plt.title('ROC Curve', fontsize = 12)
    plt.legend(loc="lower right")
    plt.show()
    
## evaluate the model on test data
def evaluate_model(model, x_test, y_test):
    # get predicted label for test set
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # get true label
    y_true = np.argmax(y_test, axis=1)

    # Calculate metrics
    cm = confusion_matrix(y_true, y_pred_classes)
    class_names = ['Class 0', 'Class 1', 'Class 2']
    acc = np.sum(y_pred_classes == y_true)/len(y_true)
    report = classification_report(y_true, y_pred_classes, target_names=class_names, digits=2)
    metrics = {'accuracy': round(acc, 2)}

    # loop over classes and compute class-specific recall, specificity and f1-score
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        recall = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = 2 * (recall * spec) / (recall + spec)
        metrics[class_name + ' recall'] = round(recall, 2)
        metrics[class_name + ' specificity'] = round(spec, 2)
        metrics[class_name + ' f1-score'] = round(f1, 2)

    
    # Return values for report (metrics is not returned but the function could be adjusted to also incorporate the metrics)
    return report

