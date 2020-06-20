import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn import svm,preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
TRAIN_PATH='dataset/train.csv'
TEST_PATH='dataset/test.csv'
df=pd.read_csv(TRAIN_PATH)
#print(df.head())

def split_dddmmyy(df):
	df["Month"]=0
	yearly=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
	for count,rows in enumerate(df.DATE):
		for count_month,months in enumerate(yearly):
			if rows.split("-")[1].split("-")[0]==months:
	
				df.Month[count]=count_month+1
	
	df['Year']=0
	for count,rows in enumerate(df.DATE):
		
			

		df.Year[count]=rows.split("-")[-1].split("-")[-1]
	for count,rows in enumerate(df.Year):
		if rows>20:
			df.Year[count]+=1900
		else:
			df.Year[count]+=2000
	df["Date"]=0
	for count,rows in enumerate(df.DATE):
		df.Date[count]=rows.split("-")[0]

			#print(rows,df.DATE[count])

	

	df.drop(["DATE"],1,inplace=True)
	#print(df.head())
	df.to_csv('checkmate.csv',index=False)
	#df.fillna(-1,inplace=True)
	return df

df=split_dddmmyy(df)
#print(df.tail())
df.drop(['INCIDENT_ID'],1,inplace=True)
x=np.array(df.drop(['MULTIPLE_OFFENSE'],1).astype(float))
#print(x)
y=np.array(df.MULTIPLE_OFFENSE)
clf=XGBClassifier(max_depth=2,colsample_bytree=0.8,colsample_bylevel=0.8)


# plot feature importance

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf.fit(x_train,y_train)
#print(clf.feature_importances_)
#plot_importance(clf)
#print(x)
#plt.show()
accuracy=clf.score(x_test,y_test)
print(accuracy)

df1=pd.read_csv(TEST_PATH)
df1=split_dddmmyy(df1)
print(df1.head())
with open("submission4.csv","w") as f:
	f.write('INCIDENT_ID,MULTIPLE_OFFENSE\n')
for Counter,rows in enumerate(df1.X_1):
	prediction=np.array([df1.X_1[Counter],df1.X_2[Counter],df1.X_3[Counter],df1.X_4[Counter],df1.X_5[Counter],df1.X_6[Counter],df1.X_7[Counter],df1.X_8[Counter],df1.X_9[Counter],df1.X_10[Counter],df1.X_11[Counter],df1.X_12[Counter],df1.X_13[Counter],df1.X_14[Counter],df1.X_15[Counter],df1.Month[Counter],df1.Year[Counter],df1.Date[Counter]])
	prediction=prediction.reshape(-1,len(prediction))
	class_=clf.predict(prediction)
	with open("submission4.csv","a") as f:
		f.write("{},{}\n".format(df1.INCIDENT_ID[Counter],class_[0]))
	
