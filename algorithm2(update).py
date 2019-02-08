model=""
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

def evaluation(df,val):
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
		model = LogisticRegression(random_state = 0)
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(X_test)
		cm = confusion_matrix(y_test, y_pred)
		return str(modelpr *100) + "<br>"+"<br>"+"<br>"+"confusion_matrix"+"<br>"+str(cm)

	elif(val == 2):
		from sklearn.neighbors import KNeighborsClassifier
		from sklearn.metrics import confusion_matrix
		model = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(X_test)
		cm = confusion_matrix(y_test, y_pred)
		return str(modelpr *100) + "<br>"+"<br>"+"<br>"+"confusion_matrix"+"<br>"+str(cm)

	elif(val == 3):
		from sklearn.naive_bayes import GaussianNB
		from sklearn.metrics import confusion_matrix
		model = GaussianNB()
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(X_test)
		cm = confusion_matrix(y_test, y_pred)
		return str(modelpr*100) + "<br>"+"<br>"+"<br>"+"confusion_matrix"+"<br>"+str(cm)

	elif(val == 4):
		from sklearn.svm import SVC
		from sklearn.metrics import confusion_matrix
		model = SVC(kernel='linear')#SVC(kernel='rbf')
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(X_test)
		cm = confusion_matrix(y_test, y_pred)
		return str(modelpr *100) + "<br>"+"<br>"+"<br>"+"confusion_matrix"+"<br>"+str(cm)

	elif(val == 5):
		from sklearn.tree import DecisionTreeClassifier
		from sklearn.metrics import confusion_matrix
		model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(X_test)
		cm = confusion_matrix(y_test, y_pred)
		return str(modelpr *100) + "<br>"+"<br>"+"<br>"+"confusion_matrix"+"<br>"+str(cm)

	elif(val == 6):
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.metrics import confusion_matrix
		model = RandomForestClassifier(n_estimators=10)
		model.fit(X_train,y_train)
		modelpr=model.score(X_test,y_test)*100
		y_pred=model.predict(X_test)
		cm = confusion_matrix(y_test, y_pred)
		return str(modelpr *100) + "<br>"+"<br>"+"<br>"+"confusion_matrix"+"<br>"+str(cm)

	else:
		return "not found"



def predData(df,data,val):
	#data split
	X = df[["Dependents","tenure","OnlineSecurity","TechSupport","Contract","PaperlessBilling","MonthlyCharges","TotalCharges"]]
	y = df['Churn']
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
	from sklearn.preprocessing import StandardScaler #Scalling
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_test = sc.transform(X_test)
	
	if(val ==1):
		from sklearn.linear_model import LogisticRegression
		from sklearn.metrics import confusion_matrix
		model = LogisticRegression(random_state = 0)
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(data)
		return str(y_pred)
	elif(val == 2):
		from sklearn.neighbors import KNeighborsClassifier
		from sklearn.metrics import confusion_matrix
		model = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(data)
		return str(y_pred)

	elif(val == 3):
		from sklearn.naive_bayes import GaussianNB
		from sklearn.metrics import confusion_matrix
		model = GaussianNB()
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(data)
		return str(y_pred)

	elif(val == 4):
		from sklearn.svm import SVC
		from sklearn.metrics import confusion_matrix
		model = SVC(kernel='linear')#SVC(kernel='rbf')
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(data)
		return str(y_pred)

	elif(val == 5):
		from sklearn.tree import DecisionTreeClassifier
		from sklearn.metrics import confusion_matrix
		model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
		model.fit(X_train, y_train)
		modelpr=model.score(X_test, y_test)
		y_pred=model.predict(data)
		return str(y_pred)

	elif(val == 6):
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.metrics import confusion_matrix
		model = RandomForestClassifier(n_estimators=10)
		model.fit(X_train,y_train)
		modelpr=model.score(X_test,y_test)*100
		y_pred=model.predict(data)
		return str(y_pred)
		
	else:
		return "not found"

	
