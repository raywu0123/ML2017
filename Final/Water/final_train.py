import pandas as pd
import numpy as np
import xgboost as xgb
import random
import pickle


status_group = ["functional", "non functional", "functional needs repair"]


#%%
##############################feature processing part##############################
train = pd.read_csv("train_data.csv")
test = pd.read_csv("test.csv")
test = test.fillna(test.median())

#column_labels = list(train.columns.values)
#column_labels.remove("id")
#column_labels.remove("date_recorded")
#column_labels.remove("status_group")
column_labels = ['amount_tsh','funder', 'gps_height', 
'longitude','latitude', 'basin', 'lga', 'population', 
'public_meeting', 'scheme_name','permit','construction_year',
'extraction_type_class','management','management_group','payment', 
'quantity', 'quality_group', 'source','source_type','source_class','waterpoint_type', 'date_recorded']


#%%
########################## train/test preprocessing ####################################
date_weight = [365,30,1]

########## date_recorded for trainig data ##########

print("start processing feature: date_recorded")

for idx, date in enumerate(train["date_recorded"]):
    train["date_recorded"][idx] = sum(np.array(date.split("-"),dtype= np.int)*date_weight)
    
min_date = np.min(np.array(train["date_recorded"]))
train["date_recorded"] = (train["date_recorded"]- min_date).astype(int)
    
print("Features date_recorded for trainig: successfully")

########## date_recorded for testing data ##########

for idx, date in enumerate(test["date_recorded"]):
    test["date_recorded"][idx] = np.sum(np.array(date.split("-"),dtype= np.int)*date_weight)
    
test["date_recorded"] = (test["date_recorded"]- min_date).astype(int)

print("Features date_recorded for testing: successfully")

#%%
for model_term in range(4):
    ##############################Training part##############################
    models = []
    models_acc = []
    
    for t in range(10):
        random_num = random.randint(1,100000)
        
        dtrain = xgb.DMatrix( train[column_labels], label=train["status_group"], missing=0)
        #dvalid = xgb.DMatrix( validation[column_labels], label=validation["status_group"], missing=0)
        #dtest = xgb.DMatrix( test[column_labels], missing=0)
        
        print("Asign train/val/test data to DMatrix: successfully")
        '''
        param = {'max_depth':23, 
                 'eta':0.3, 
                 'silent':0, 
                 'subsample':0.7,
                 'seed':random_num,       
                 'objective':'multi:softprob', 
                 'num_class':3 }
        
        watchlist  = [(dvalid,'eval'), (dtrain,'train')]
        num_round = 30
            
        bst = xgb.train(param, dtrain, num_round, watchlist) 
        '''
        
        clf = xgb.XGBClassifier(n_estimators = 500,
            learning_rate = 0.2,
            objective = 'multi:softmax',
            booster = 'gbtree',
            colsample_bytree = 0.4,
            random_state = random_num)
            
        xgb_params = clf.get_xgb_params()
        xgb_params['num_class'] = 3
        xgb_params['max_depth'] = 12
            
        cv_result = xgb.cv(xgb_params,
        dtrain,
        num_boost_round = 1000,
        nfold = 5,
        metrics = {"merror"},
        maximize = False,
        early_stopping_rounds = 10,
        seed = random_num,
        callbacks = [xgb.callback.print_evaluation(show_stdv = False)]
        )
        
        print("finish cv")
        clf.set_params(max_depth = 14)
        clf.set_params(n_estimators = cv_result.shape[0])
        clf.fit(train[column_labels], train["status_group"], eval_metric= "merror")
            
        # Accuracy
        # accuracy= np.sum(clf.predict(validation[column_labels])==validation["status_group"]) / validation.shape[0]
        # print("Train Accuracy= " , accuracy , "%")
        models.append(clf)
        models_acc.append(cv_result.iloc[cv_result.shape[0]-1]["test-merror-mean"])
        print("finish ", t, "st time.")
    
    
    ##############################Testing part##############################
    models_acc = np.array(models_acc)
    best_model = models[np.argmin(models_acc)]
    
    ########## Predict ##########
    
    #result = best_model.predict(test[column_labels])
    #print("Prediction for test data: successfully")
    
    '''
    result=[]
    
    for prob in result_prob:
        result.append(np.argmax(prob))
    '''
    
    
    '''
    ### Making submission file###
    # Dataframe as per submission format
    submission = pd.DataFrame({
    			"id": test["id"],
    			"status_group": result
    		})
    for i in range(len(status_group)):
    	submission.loc[submission["status_group"] == i, "status_group"] = status_group[i]
    print("Dataframe as per submission format: successfully")
    
    # Store submission dataframe into file
    submission.to_csv("submission.csv", index = False)
    print("Store submission dataframe into file: successfully")
    '''
    
    pickle.dump(best_model, open("models/model_"+ str(model_term) +".dat", "wb"))
    print("finish writing model")




'''
how to load model?

loaded_model = pickle.load(open("models/model_840.dat", "rb"))
y_pred = loaded_model.predict("...")
'''

###################### Ensemble ###############################
#%%
model_xgb_1 = pickle.load(open("models/model_0.dat", "rb"))
model_xgb_2 = pickle.load(open("models/model_1.dat", "rb"))
model_xgb_3 = pickle.load(open("models/model_2.dat", "rb"))
model_xgb_4 = pickle.load(open("models/model_3.dat", "rb"))
model_xgb_5 = pickle.load(open("models/model_4.dat", "rb"))
model_xgb_6 = pickle.load(open("models/model_840.dat", "rb"))

result_prob_1 = model_xgb_1.predict_proba(test[column_labels])
result_prob_2 = model_xgb_2.predict_proba(test[column_labels])
result_prob_3 = model_xgb_3.predict_proba(test[column_labels])
result_prob_4 = model_xgb_4.predict_proba(test[column_labels])
result_prob_5 = model_xgb_5.predict_proba(test[column_labels])
result_prob_6 = model_xgb_6.predict_proba(test[column_labels])

final_result = (result_prob_1*result_prob_2*result_prob_3*result_prob_4*result_prob_5*result_prob_6) ** (1/6)

xgb_ensemble_result = []
for i in range(final_result.shape[0]):
    xgb_ensemble_result.append(np.argmax(final_result[i]))
    
#%%
#with random forest

import gc
import pickle
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

column_labels_rf = list(train.columns.values)
column_labels_rf.remove("id")
column_labels_rf.remove("date_recorded")
column_labels_rf.remove("status_group")

clf_1 = joblib.load('rf_models/clf_1.pkl','r')
clf_2 = joblib.load('rf_models/clf_2.pkl','r')
clf_3 = joblib.load('rf_models/clf_3.pkl','r')
clf_4 = joblib.load('rf_models/clf_4.pkl','r')
clf_5 = joblib.load('rf_models/clf_5.pkl','r')

prediction_prob_1 = clf_1.predict_proba(test[column_labels_rf])
prediction_prob_2 = clf_2.predict_proba(test[column_labels_rf])
prediction_prob_3 = clf_3.predict_proba(test[column_labels_rf])
prediction_prob_4 = clf_4.predict_proba(test[column_labels_rf])
prediction_prob_5 = clf_5.predict_proba(test[column_labels_rf])

final_result2 = (result_prob_1*result_prob_2*result_prob_3*result_prob_4*result_prob_5*result_prob_6
                *prediction_prob_1*prediction_prob_2*prediction_prob_3*prediction_prob_4*prediction_prob_5) ** (1/11)

xgb_rf_ensemble_result = []
for i in range(final_result.shape[0]):
    xgb_rf_ensemble_result.append(np.argmax(final_result2[i]))

#%%
### Making submission file###
# Dataframe as per submission format
submission = pd.DataFrame({
			"id": test["id"],
			"status_group": xgb_rf_ensemble_result
		})
for i in range(len(status_group)):
	submission.loc[submission["status_group"] == i, "status_group"] = status_group[i]
print("Dataframe as per submission format: successfully")

# Store submission dataframe into file
submission.to_csv("xgb_rf_geomean_soft_ensemble.csv", index = False)
print("Store submission dataframe into file: successfully")


#%%
########################## Hard Label #############################

model_xgb_1 = pickle.load(open("models/model_0.dat", "rb"))
model_xgb_2 = pickle.load(open("models/model_1.dat", "rb"))
model_xgb_3 = pickle.load(open("models/model_2.dat", "rb"))
model_xgb_4 = pickle.load(open("models/model_3.dat", "rb"))
model_xgb_5 = pickle.load(open("models/model_4.dat", "rb"))
model_xgb_6 = pickle.load(open("models/model_840.dat", "rb"))

result_1 = model_xgb_1.predict(test[column_labels])
result_2 = model_xgb_2.predict(test[column_labels])
result_3 = model_xgb_3.predict(test[column_labels])
result_4 = model_xgb_4.predict(test[column_labels])
result_5 = model_xgb_5.predict(test[column_labels])
result_6 = model_xgb_6.predict(test[column_labels])

#with random forest
import gc
import pickle
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

column_labels_rf = list(train.columns.values)
column_labels_rf.remove("id")
column_labels_rf.remove("date_recorded")
column_labels_rf.remove("status_group")

clf_1 = joblib.load('rf_models/clf_1.pkl','r')
clf_2 = joblib.load('rf_models/clf_2.pkl','r')
clf_3 = joblib.load('rf_models/clf_3.pkl','r')
clf_4 = joblib.load('rf_models/clf_4.pkl','r')
clf_5 = joblib.load('rf_models/clf_5.pkl','r')

prediction_1 = clf_1.predict(test[column_labels_rf])
prediction_2 = clf_2.predict(test[column_labels_rf])
prediction_3 = clf_3.predict(test[column_labels_rf])
prediction_4 = clf_4.predict(test[column_labels_rf])
prediction_5 = clf_5.predict(test[column_labels_rf])


hard_label_final = np.zeros((14850,3), dtype= int)
for i in range(14850):
    hard_label_final[i, result_1[i]] += 1
    hard_label_final[i, result_2[i]] += 1
    hard_label_final[i, result_3[i]] += 1
    hard_label_final[i, result_4[i]] += 1
    hard_label_final[i, result_5[i]] += 1
    hard_label_final[i, result_6[i]] += 1
    hard_label_final[i, prediction_1[i]] += 1
    hard_label_final[i, prediction_2[i]] += 1
    hard_label_final[i, prediction_3[i]] += 1
    hard_label_final[i, prediction_4[i]] += 1
    hard_label_final[i, prediction_5[i]] += 1

xgb_rf_hard_ensemble_result = []
for i in range(final_result.shape[0]):
    xgb_rf_hard_ensemble_result.append(np.argmax(hard_label_final[i]))
#%%
### Making submission file###
# Dataframe as per submission format
submission = pd.DataFrame({
			"id": test["id"],
			"status_group": xgb_rf_hard_ensemble_result
		})
for i in range(len(status_group)):
	submission.loc[submission["status_group"] == i, "status_group"] = status_group[i]
print("Dataframe as per submission format: successfully")

# Store submission dataframe into file
submission.to_csv("xgb_rf_hard_ensemble.csv", index = False)
print("Store submission dataframe into file: successfully")



