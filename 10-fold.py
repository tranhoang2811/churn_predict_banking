import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(8, 8))
models = ['KNN', 'LR', 'DT', 'RF', 'SVM']
X_axis = np.arange(len(models))

cluster_1 = {
    'accuracies': [90.6, 88.21, 93.87, 95.83, 92.98],
}

cluster_2 = {
    'accuracies': [90.89, 91.76, 94.71, 96.98, 93.88],
}

cluster_3 = {
    'accuracies': [92.58, 84.89, 93.49, 98.49, 97.1],
}

cluster_4 = {
    'accuracies': [88.76, 83.51, 91.6, 95.72, 93.25],
}

cluster_5 = {
    'accuracies': [92.99, 85.98, 93.48, 97.8, 95.06],
}

cluster_6 = {
    'accuracies': [91.7, 89.29, 95.23, 98.66, 97.46],
}


accuracy_label = plt.bar(X_axis, cluster_3['accuracies'], width=0.6, color='#0b84a5', label='Accuracy')

plt.bar_label(accuracy_label, padding=1, fontsize=12)

plt.xlabel('Model', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Cluster III 10-fold results', fontsize=16)
plt.xticks(X_axis, models)
plt.show()
