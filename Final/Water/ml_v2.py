author='ray'

discrete_features=['funder','installer','wpt-name','basin','subvilliage',
'region','lga','ward','recorded_by','scheme_management','scheme_name',
'extraction_type','extracton_type_group','extraction_type_class','management',
'management_group','payment','payment_type','water_quality','quality_group',
'quantity','quantity_group','source,source_type','source_class','waterpoint_type',
'waterpoint_type_group']

import gc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.metrics import accuracy_score
import time
from keras.models import Sequential,Model
from keras.layers import Dense,Embedding,Input,Flatten,Concatenate
from keras.utils import np_utils
status_group = ["functional", "non functional", "functional needs repair"]

def get_data(train_data_path,test_data_path):
	###Training part###
	# Traning data
	train = pd.read_csv("train_data.csv")
	print("Traning data: successfully")
	# Testing data
	test = pd.read_csv("test.csv")
	test = test.fillna(test.median())
	print("Testing data: successfully")
	# Features Selection
	column_labels = list(train.columns.values)

	column_labels.remove("id")
	column_labels.remove("date_recorded")
	column_labels.remove("status_group")
	column_labels.remove('region')## region code instead
	column_labels.remove("recorded_by")## all same
	column_labels.remove("extraction_type_class")## extraction type instead
	column_labels.remove("management_group")## management instead
	column_labels.remove("payment_type")## payment instead
	column_labels.remove("quality_group")## water quality instead
	column_labels.remove("source_type")## source instead
	column_labels.remove("source_class")## source instead
	column_labels.remove("waterpoint_type_group")## watertype instead

	selected_discrete_features=[]
	selected_continuous_features=[]
	for feature in column_labels:
		if feature in discrete_features:
			selected_discrete_features.append(feature)
		else:
			selected_continuous_features.append(feature)
	print('discete:',selected_discrete_features)
	print('continuous:',selected_continuous_features)
	print('total num of feats:',len(selected_discrete_features)+len(selected_continuous_features))

	print("Features for trainig: successfully")
	return train,test,selected_continuous_features,selected_discrete_features

def training(train,selected_continous_feats,selected_discrete_feats):

	column_labels=selected_continous_feats+selected_discrete_feats
	# Assign data for validation
	amount = int(0.9 * len(train))
	validation = train[amount:]
	train=train.sample(frac=1)
	train = train[:amount]
	print("Assign data for validation: successfully")
	train_x=np.asarray(train[column_labels])
	train_y=np.asarray(train["status_group"])
	train_y_class=np_utils.to_categorical(train_y,3)
	print(train_y.shape)
	print(train_x.shape)
	train_x_c=np.asarray(train[selected_continous_feats],dtype=float)
	train_x_d=np.asarray(train[selected_discrete_feats],dtype=int)


	def random_forest_model(train_x,train_y):
		# Classifier
		clf = RandomForestClassifier(n_estimators=230, n_jobs=-1,class_weight='balanced')
		print("Classifier: successfully")
		# Traning
		clf.fit(train_x,train_y)
		print("Traning: successfully")
		# Accuracy
		accuracy = accuracy_score(clf.predict(validation[column_labels]), validation["status_group"])
		print("Accuracy = " + str(accuracy))
		print("Accuracy: successfully")
		return clf
	def adaboost_model(train_x,train_y):
		# Classifier
		clf = AdaBoostClassifier(n_estimators=300)
		print("Classifier: successfully")
		# Traning
		clf.fit(train_x,train_y)
		print("Traning: successfully")
		# Accuracy
		accuracy = accuracy_score(clf.predict(validation[column_labels]), validation["status_group"])
		print("Accuracy = " + str(accuracy))
		return clf
	def bagging_model(train_x,train_y):
		# Classifier
		clf = BaggingClassifier(n_estimators=50)
		print("Classifier: successfully")
		# Traning
		clf.fit(train_x,train_y)
		print("Traning: successfully")
		# Accuracy
		accuracy = accuracy_score(clf.predict(validation[column_labels]), validation["status_group"])
		print("Accuracy = " + str(accuracy))
		return clf
	def keras_model(train_x_c,train_x_d,train_y_class):
		embedding_dim=50
		c_inputs=[]
		d_inputs=[]
		emb_vecs=[]
		for feature in train_x_c.T:
			c_inputs.append(Input(shape=[1]))
		for feature in train_x_d.T:
			d_inputs.append(Input(shape=[1]))
			emb_vec=Embedding(np.max(feature)+1,embedding_dim)(d_inputs[-1])
			emb_vec=Flatten()(emb_vec)
			emb_vecs.append(emb_vec)
		merge=Concatenate()(c_inputs+emb_vecs)
		merge=Dense(500,activation='relu')(merge)
		merge=Dense(500,activation='relu')(merge)
		merge=Dense(3,activation='softmax')(merge)

		model=Model(c_inputs+d_inputs,merge)
		print(model.summary())
		model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
		print(train_x_c.shape)
		print(train_x_d.shape)
		train_x_full=[]
		for feature in np.concatenate((train_x_c,train_x_d),axis=1).T:
			train_x_full.append(feature)
		print(len(train_x_full))
		print(train_x_full[0].shape)
		model.fit(train_x_full,train_y_class,batch_size=100,epochs=10,validation_split=0.1)
		return model
	#clf=random_forest_model(train_x,train_y)
	#clf=bagging_model(train_x,train_y)
	clf=keras_model(train_x_c,train_x_d,train_y_class)
	#clf=adaboost_model(train_x,train_y)
	return clf

def testing(test,selected_continous_feats,selected_discrete_feats,clf):
	###Testing part###
	column_labels=selected_continous_feats+selected_discrete_feats
	# Prediction for test data
	prediction = clf.predict(test[column_labels])
	print("Prediction for test data: successfully")
	### Making submission file###
	# Dataframe as per submission format
	submission = pd.DataFrame({"id": test["id"], "status_group": prediction})
	for i in range(len(status_group)):
		submission.loc[submission["status_group"] == i, "status_group"] = status_group[i]
	print("Dataframe as per submission format: successfully")

	# Store submission dataframe into file
	submission.to_csv("submission"+str(time.asctime())+".csv", index=False)
	print("Store submission dataframe into file: successfully")

def main():
	train,test,selected_continuous_feats,selected_discrete_feats=get_data('train_data.csv','test.csv')
	clf=training(train,selected_continuous_feats,selected_discrete_feats)

	testing(test,selected_continuous_feats,selected_discrete_feats,clf)

if __name__=='__main__':
	main()