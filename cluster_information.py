import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
credit_data = pd.read_csv('./dataset/original/valid_data_test.csv')
print(f'Data shape {credit_data.shape}')

quantitative_label = ['Dependent_count', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
                      'Total_Amt_Chng_Q4_Q1', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']
quantitative_data = []
Y_axis = np.arange(start=1, stop=len(quantitative_label) + 1)


for label in quantitative_label:
    quantitative_data.append(credit_data[label].values)

print(f'Data values: {quantitative_data}')
plt.figure(figsize=(14, 10))
plt.boxplot(quantitative_data, patch_artist=True, vert=False)
plt.title('Quantitative data chart', fontsize=12)
plt.xlabel('Values', fontsize=12)
plt.ylabel('Label')
plt.yticks(Y_axis, quantitative_label, fontsize=10)
plt.show()

