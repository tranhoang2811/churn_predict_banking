from imblearn.over_sampling import SMOTE
import pandas as pd

clusters_data = []
for k in range(6):
    clusters_data.append(pd.read_csv(f'../dataset/before_smote/cluster_{k+1}.csv'))

clusters_with_balance_data = []
for cluster in clusters_data:
    y = cluster['Churn_Label']
    print(y.value_counts())
    print(y.shape)
    X = cluster.drop(columns='Churn_Label')
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X, y)
    print(y_smote.value_counts())
    balance_data = pd.concat([X_smote, y_smote], axis=1)
    clusters_with_balance_data.append(balance_data)
    print('*'*20)

# for k in range(6):
#     balance_data = clusters_with_balance_data[k]
#     balance_data.to_csv(f'../dataset/after_smote/cluster_{k + 1}_with_balance.csv', index=False)
