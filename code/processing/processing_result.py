import matplotlib.pyplot  as plt
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_auc_score, roc_curve, accuracy_score
import pandas as pd
import numpy as np

def plot_accuracy_loss(history,path_accuracy_png,path_loss_png):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path_accuracy_png)
    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path_loss_png)

def plot_precision_recall_curve(Y_pred,trainY,testY,path_auprc_plot,label):
    plt.figure()
    # retrieve just the probabilities for the positive class
    pos_probs = Y_pred[:, 1]
    # calculate the no skill line as the proportion of the positive class
    y = np.append(trainY[:,1], testY[:,1], axis=0)
    no_skill = len(y[y == 1]) / len(y)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # calculate model precision-recall curve

    precision, recall, _ = precision_recall_curve(testY[:,1], pos_probs)
    auc_score = auc(recall, precision)
    print('Logistic PR AUC: %.3f' % auc_score)
    # plot the model precision-recall curve
    plt.plot(recall, precision, marker='.', label=label)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    plt.savefig(path_auprc_plot)




def classification_report_csv(report,evaluation_file):
    report_data = []
    lines = report.split('\n')
    lines = [t for t in lines if len(t) > 1]
    for line in lines[1:-3]:
        row = {}
        row_data = line.split('      ')
        row_data = [t for t in row_data if len(t) > 1]
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(evaluation_file,mode='a', index = False)
    return report_data

def print_results(y_test,y_pred, y_pred_non_bin):
    print('accuracy %s' % accuracy_score(y_test, y_pred))
    precision, recall, _ = precision_recall_curve(y_test, y_pred_non_bin)
    auc_score = auc(recall, precision)
    print('PR AUC: %.3f' % auc_score)
    # calculate roc auc
    roc_auc = roc_auc_score(y_test, y_pred_non_bin)
    print('ROC AUC %.3f' % roc_auc)

def save_results(evaluation_file,y_test,y_pred, y_pred_non_bin):
    print('accuracy %s' % accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    report_data = classification_report_csv(report, evaluation_file)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_non_bin)
    auc_score = auc(recall, precision)
    print('PR AUC: %.3f' % auc_score)
    # calculate roc auc
    roc_auc = roc_auc_score(y_test, y_pred_non_bin)
    print('ROC AUC %.3f' % roc_auc)
    # print("c=",c)
    with open(evaluation_file, 'a') as f:
        f.write('accuracy, %s\n' % accuracy_score(y_test, y_pred))
        f.write('auprc, %s\n' % auc_score)
        f.write('auc, %s\n' % roc_auc)
    return report_data

def plot_auc_curve(Y_pred,testY,path_auc_file,label):
    pos_probs = Y_pred[:, 1]
    plt.figure()
    # plot no skill roc curve
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    # calculate roc curve for model
    fpr, tpr, _ = roc_curve(testY[:,1], pos_probs)
    # plot model roc curve
    plt.plot(fpr, tpr, marker='.', label=label)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.savefig(path_auc_file)
