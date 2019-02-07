import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from flask import Flask,jsonify,request,render_template, jsonify
import algorithm
from sklearn.preprocessing import LabelEncoder
from algorithm import predict
#classifier  = predict(df,val)


df=pd.read_csv("Dataset.csv")
app = Flask(__name__)

#@app.route('/')
#def p():
#	return df.head().to_html()


	

@app.route('/')
def ab():
	return "<form action='/prep' method='get'><input type='submit' value='perprocessing'></form>"


@app.route('/prep')
def prep():
	
	return algorithm.perprocess(df).to_html()

@app.route('/showpred')
def show():
	return "<h1>select your algorithm</h1>"+"<form action='/predict' method='get'><select name='pr'>\
	<option value='1'>Logistic Regression</option>\
	<option value='2'>K N N</option>\
	<option value='3'>Naive Bayes</option>\
	<option value='4'>S V M</option>\
	<option value='5'>Decision Tree</option>\
	<option value='6'>Random Forest</option>\
	<input type='submit' value='perprocessing'></form>"

@app.route('/predict')
def pred():
	pr=int(request.args["pr"])
	return "Accuracy  "+algorithm.predict(algorithm.perprocess(df),pr) +"<a href=adddata>"+"<br>"+"<br>"+"<br>"+"<br>"+"<input type=button value='go to predict'/adddata/'></a>"



                                                                                 








@app.route('/adddata')
def add():
	return "<h1>add your data</h1>"+"<form action='/about' method='get' >\
	Dependents      \t <select name='dependence'><option value='1'>yes</option><option value='0'>No</option></select><br>\
	tenure          \t <input name='tenure' placeholder='tenure' type 'text'><br>\
	OnlineSecurity  \t <select name='OnlineSecurity'><option value='1'>yes</option><option value='0'>No</option><option value='0'>No internet service</option></select><br>\
	TechSupport     \t <select name='TechSupport'><option value='1'>yes</option><option value='0'>No</option><option value='0'>No internet service </option></select><br>\
	Contract        \t<select name='Contract'><option value='0'>month to month</option><option value='1'>one year</option><option value='2'>two year</option></select> <br>\
	PaperlessBilling\t <select name='PaperlessBilling'><option value='1'>yes</option><option value='0'>No</option></select><br>\
	MonthlyCharges  \t <input name='MonthlyCharges' placeholder='tenure' type 'text'><br>\
	TotalCharges    \t <input name='TotalCharges' placeholder='tenure' type 'text'><br><br><input type='submit' value='predict' >"

@app.route('/about')
def about():
	Dependents      =int(request.args["dependence"])
	tenure          =float(request.args["tenure"] )
	OnlineSecurity  =int(request.args["OnlineSecurity"])
	TechSupport     =int(request.args["TechSupport"] )
	Contract	    =int(request.args["Contract"] )
	PaperlessBilling=int(request.args["PaperlessBilling"])
	MonthlyCharges  =float(request.args["MonthlyCharges"])
	TotalCharges    =float(request.args["TotalCharges"])
	data =[Dependents,tenure,OnlineSecurity,TechSupport,Contract,PaperlessBilling,MonthlyCharges,TotalCharges]
	newdata = pd.DataFrame(data)
	#d={}
	#c=0
	#for i in dfnew:
	#	d[i]=[data[c]]
	#	c+=1
	#dataa=pd.DataFrame(dataa=d)
	#from sklearn.neighbors import KNeighborsClassifier
	#knnclassifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2)
	
	#Chrn = clsf.predict(newdata,val)
    #return Chrn

	return newdata.to_html()
	


#@app.route('/churn')
#def chrn():	
#	return  chrn(newdata)


    










if __name__ == '__main__':
    app.run(debug=True,host="127.0.0.1",port=5000)