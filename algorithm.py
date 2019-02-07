
def perprocess(df):
	cols = ["Dependents","tenure","OnlineSecurity","TechSupport","Contract","PaperlessBilling","MonthlyCharges","TotalCharges","Churn"]
	dfnew=df.copy()
	dfnew=dfnew[cols]
	tenure_mean = dfnew['tenure'].mean()
	fills = {'tenure':tenure_mean }
	dfnew.fillna(fills , inplace = True)
	return lencoding(dfnew) 

def lencoding(dfnew):
	
	from sklearn.preprocessing import LabelEncoder	
	lb_Dependents=LabelEncoder()
	dfnew['Dependents']=lb_Dependents.fit_transform(dfnew['Dependents'])
	lb_OnlineSecurity=LabelEncoder()
	dfnew['OnlineSecurity']=lb_OnlineSecurity.fit_transform(dfnew['OnlineSecurity'])	
	lb_TechSupport=LabelEncoder()
	dfnew['TechSupport']=lb_TechSupport.fit_transform(dfnew['TechSupport'])	
	lb_Contract=LabelEncoder()
	dfnew['Contract']=lb_Contract.fit_transform(dfnew['Contract'])	
	lb_PaperlessBilling=LabelEncoder()
	dfnew['PaperlessBilling']=lb_PaperlessBilling.fit_transform(dfnew['PaperlessBilling'])	
	lb_Churn=LabelEncoder()
	dfnew['Churn']=lb_Churn.fit_transform(dfnew['Churn'])
	return standardization(dfnew)

def standardization(dfnew):


	dfnew["tenure"]=(dfnew["tenure"]-dfnew["tenure"].mean())/dfnew["tenure"].std() #standarization z-score=(X-mean)/std
	dfnew["MonthlyCharges"]=(dfnew["MonthlyCharges"]-dfnew["MonthlyCharges"].mean())/dfnew["MonthlyCharges"].std()
	dfnew["TotalCharges"]=(dfnew["TotalCharges"]-dfnew["TotalCharges"].mean())/dfnew["TotalCharges"].std()
	return dfnew

#LogisticRegression
def predict(df,val):
	#data split
	X = df.iloc[:, 0:7].values
	y = df.iloc[:, 8].values
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
	from sklearn.preprocessing import StandardScaler #Scalling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	
	
	
	if(val ==1):
		from sklearn.linear_model import LogisticRegression
		from sklearn.metrics import confusion_matrix
		LRclassifier = LogisticRegression(random_state = 0)
		LRclassifier.fit(X_train, y_train)
		LR_predict=LRclassifier.score(X_test, y_test)
		#y_pred=LRclassifier.predict(X_test)
		#cm = confusion_matrix(y_test, y_pred)
		return str(LR_predict) #str(cm)
	elif(val == 2):
		from sklearn.neighbors import KNeighborsClassifier
		knnclassifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
		knnclassifier.fit(X_train, y_train)
		KNN_predict=knnclassifier.score(X_test, y_test)
		return str(KNN_predict)
	elif(val == 3):
		from sklearn.naive_bayes import GaussianNB
		NBclassifier = GaussianNB()
		NBclassifier.fit(X_train, y_train)
		NB_predict=NBclassifier.score(X_test, y_test)
		return str(NB_predict)
	elif(val == 4):
		from sklearn.svm import SVC
		SVMclassifier = SVC(kernel='linear')#SVC(kernel='rbf')
		SVMclassifier.fit(X_train, y_train)
		SVM_predict=SVMclassifier.score(X_test, y_test)
		return str(SVM_predict)
	elif(val == 5):
		from sklearn.tree import DecisionTreeClassifier
		DTclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
		DTclassifier.fit(X_train, y_train)
		DT_predict=DTclassifier.score(X_test, y_test)
		return str(DT_predict)

	else:
		return "not found"
	
