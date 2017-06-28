import pandas as pd

pre_train_label = pd.read_csv("pre_train_label.csv")
pre_train_value=pd.read_csv("pre_train_value.csv")
pre_train_value['status_group']=pre_train_label['status_group']
pre_train_value=pre_train_value.sort_index(by='id')
print(pre_train_value)
pre_train_value.to_csv("train_data.csv", index = False)