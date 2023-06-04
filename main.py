import numpy as np
import matplotlib.pyplot as plt

# plt.figure(figsize=(9, 9))
#
# models = ['KNN', 'LR', 'DT', 'RF', 'SVM']
#
# cluster_average_result_precision = [84.86, 87.68, 92.09, 96.56, 96.63]
# cluster_average_result_recall = [98.81, 87.85, 93.98, 97.45, 90.35]
# cluster_average_result_f1_score = [91.28, 87.76, 93, 97, 93.36]
#
# X_axis = np.arange(len(models))
#
# sample_result_label = plt.bar(X_axis, cluster_average_result_f1_score, color='#0b84a5')
#
# plt.xticks(X_axis, models)
# plt.bar_label(sample_result_label, padding=2, fontsize=8)
#
# plt.xlabel('Model', fontsize=12)
# plt.ylabel('F1 score', fontsize=12)
# plt.title('The cluster average F1 scores with hold out', fontsize=12)
#
# plt.show()

# ======================================================================================================================

plt.figure(figsize=(9, 9))

models = ['KNN', 'LR', 'DT', 'RF', 'SVM']
# For 10 fold
# sample_result = [90.93, 86.35, 94.4, 97.36, 95.27]
# cluster_average_result = [91.25, 87.27, 93.73, 97.25, 94.96]

# For hold out
# sample_result = [89.58, 86.26, 94.05, 97.4, 94.19]
# cluster_average_result = [90.35, 87.57, 92.83, 97, 93.66]

# For precision
# sample_result = [83.08, 85.78, 93.03, 96.92, 96.15]
# cluster_average_result = [84.86, 87.68, 92.09, 96.56, 96.63]

# For recall
sample_result = [99.15, 87.11, 94.99, 97.94, 91.73]
cluster_average_result = [98.81, 87.85, 93.98, 97.45, 90.35]

# For F1 score
# sample_result = [90.41, 86.44, 94, 97.43, 93.89]
# cluster_average_result = [91.28, 87.76, 93, 97, 93.36]
X_axis = np.arange(len(models))

sample_result_label = plt.bar(X_axis-0.15, sample_result, width=0.3, color='#0b84a5', label='Sample', hatch='xx')
cluster_average_result_label = plt.bar(X_axis+0.15, cluster_average_result, color='#f6c95f', width=0.3,
                                       label='Cluster average', hatch='oo')

plt.xticks(X_axis, models)
plt.bar_label(sample_result_label, padding=2, fontsize=8)
plt.bar_label(cluster_average_result_label, padding=2, fontsize=8)

plt.xlabel('Model', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.title('The sample and cluster average recall with hold out', fontsize=12)

plt.legend(ncol=2, bbox_to_anchor=(0,0,0.6,1))
plt.show()
