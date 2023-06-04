from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import pandas as pd
import numpy as np
from time import time


credit_data = pd.read_csv('../../dataset/after_smote/balance_data_without_cluster.csv')
y_original = credit_data['Churn_Label']
X_original = credit_data.drop(columns='Churn_Label')


def hold_out(data, label):
    start_time = time()
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.3)
    classifier = SVC(C=10, gamma=1)
    classifier.fit(data_train, label_train)
    label_predict = classifier.predict(data_test)

    accuracy = round(accuracy_score(label_test, label_predict)*100, 2)
    precision = round(precision_score(label_test, label_predict)*100, 2)
    recall = round(recall_score(label_test, label_predict)*100, 2)
    f1 = round(f1_score(label_test, label_predict)*100, 2)
    confusion = confusion_matrix(label_test, label_predict)
    metric_results = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
    print('Confusion matrix:\n', confusion)
    print('Metric results: ', metric_results)
    end_time = time()
    time_of_execution = round(end_time - start_time, 2)
    print('Execute time: ', time_of_execution)
    return metric_results['Accuracy']


def hold_out_with_cluster():
    cluster_accuracy = []
    for k in range(6):
        print(f'Cluster {k+1} testing')
        cluster = pd.read_csv(f'../../dataset/after_smote/cluster_{k + 1}_with_balance.csv')
        label_cluster = cluster['Churn_Label']
        data_cluster = cluster.drop(columns='Churn_Label')
        cluster_accuracy.append(hold_out(data_cluster, label_cluster))
        print('--'*50)
    print(f'Mean accuracy {round(np.mean(cluster_accuracy), 2)}')


hold_out(X_original, y_original)
hold_out_with_cluster()


def stratified_k_fold(data, label):
    start_time = time()
    stratified = StratifiedKFold(n_splits=10, shuffle=True)
    classifier = SVC(C=10, gamma=1)
    cross_validation_result = cross_val_score(classifier, data, label, cv=stratified)
    mean = round(cross_validation_result.mean()*100, 2)
    standard_deviation = round(cross_validation_result.std()*100, 2)
    end_time = time()
    time_of_execution = round(end_time - start_time, 2)
    print('Execute time: ', time_of_execution)
    print(f'Mean: {mean} ; Standard deviation: {standard_deviation}')
    return mean


def stratified_k_fold_with_clustered_data():
    cluster_accuracy = []
    for k in range(6):
        print(f'Cluster {k + 1} testing')
        cluster = pd.read_csv(f'../../dataset/after_smote/cluster_{k + 1}_with_balance.csv')
        label_cluster = cluster['Churn_Label']
        data_cluster = cluster.drop(columns='Churn_Label')
        cluster_accuracy.append(stratified_k_fold(data_cluster, label_cluster))
        print('--' * 50)
    print(f'Mean accuracy {round(np.mean(cluster_accuracy), 2)}')


# stratified_k_fold(X_original, y_original)
stratified_k_fold_with_clustered_data()

