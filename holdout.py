import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(14, 10))
bar_width = 0.15
models = ['KNN', 'LR', 'DT', 'RF', 'SVM']
X_axis = np.arange(len(models))

cluster_1 = {
    'accuracies': [89.68, 88.69, 92.66, 95.24, 93.06],
    'precisions': [83.78, 88.24, 93.68, 94.47, 96.61],
    'recalls': [98.41, 89.29, 91.86, 95.98, 89.41],
    'f1_scores': [90.51, 88.76, 92.76, 95.22, 92.87]
}

cluster_2 = {
    'accuracies': [89.8, 90.81, 93.45, 97.48, 91.44],
    'precisions': [84.08, 90.72, 92.54, 97.18, 94.97],
    'recalls': [98.51, 90.49, 94.42, 97.68, 86.29],
    'f1_scores': [90.72, 90.6, 93.47, 97.43, 90.42]
}

cluster_3 = {
    'accuracies': [92.83, 84.59, 91.76, 98.57, 95.7],
    'precisions': [88.89, 85.17, 89.57, 98.56, 97.45],
    'recalls': [98.63, 85.17, 93.61, 98.56, 94.01],
    'f1_scores': [93.51, 85.17, 91.54, 98.56, 95.7]
}

cluster_4 = {
    'accuracies': [87.52, 84.67, 90.02, 93.76, 91.44],
    'precisions': [81.14, 85.07, 88.49, 92.63, 96.33],
    'recalls': [98.61, 85.07, 92.76, 94.96, 85.82],
    'f1_scores': [89.03, 85.07, 90.57, 93.78, 90.77]
}

cluster_5 = {
    'accuracies': [92.68, 86.18, 95.33, 98.17, 93.9],
    'precisions': [88.18, 86.83, 95.98, 98.04, 97.02],
    'recalls': [99.62, 85.43, 94.84, 98.43, 90.84],
    'f1_scores': [93.55, 86.12, 95.41, 98.23, 93.83]
}

cluster_6 = {
    'accuracies': [89.6, 90.49, 93.76, 98.81, 96.43],
    'precisions': [83.08, 90.06, 92.25, 98.46, 97.4],
    'recalls': [99.1, 91.62, 96.37, 99.07, 95.74],
    'f1_scores': [90.38, 90.83, 94.26, 98.77, 96.56]
}


accuracy_label = plt.bar(X_axis-0.225, cluster_6['accuracies'], width=bar_width, color='#0b84a5', label='Accuracy', hatch='xx')
precision_label = plt.bar(X_axis-0.075, cluster_6['precisions'], width=bar_width, color='#f6c95f', label='Precision', hatch='oo')
recall_label = plt.bar(X_axis+0.075, cluster_6['recalls'], width=bar_width, color='#6f4e7c', label='Recall', hatch='++')
f1_label = plt.bar(X_axis+0.225, cluster_6['f1_scores'], width=bar_width, color='#9dd866', label='F1 score', hatch='**')

plt.bar_label(accuracy_label, padding=2, fontsize=6)
plt.bar_label(precision_label, padding=2, fontsize=6)
plt.bar_label(recall_label, padding=2, fontsize=6)
plt.bar_label(f1_label, padding=2, fontsize=6)

plt.xlabel('Model', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.title('Cluster VI hold out results', fontsize=16)
plt.xticks(X_axis, models)
plt.legend(loc='upper left', bbox_to_anchor=(0.15, 1.009), ncol=4)
plt.show()
