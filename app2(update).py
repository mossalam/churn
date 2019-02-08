from flask import Flask,jsonify,request,render_template
app = Flask(__name__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import algorithm
from sklearn.preprocessing import LabelEncoder
from algorithm import evaluation



#df=pd.read_csv("Dataset.csv")


#@app.route('/')
#def p():
#	return df.head().to_html()


	

@app.route('/')
def ab():
	import pandas as pd
	import numpy as np
	import matplotlib.pyplot as plt
	import seaborn as sb
	import algorithm
	from sklearn.preprocessing import LabelEncoder
	from algorithm import evaluation

	return "<form action='/prep' method='get'><input type='submit' value='perprocessing'></form>" +"<a href=showpred>"+"<br>"+"<br>"+"<br>"+"<br>"+"<input type=button value=' prediction evaluation'/showpred/'></a>"+"<a href=adddata>"+"<br>"+"<br>"+"<br>"+"<br>"+"<input type=button value=' predict value'/adddata/'></a>"

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
	<input type='submit' value='evaluat'></form>"

@app.route('/predict')
def pred():
	pr=int(request.args["pr"])
	return "Accuracy  "+algorithm.evaluation(algorithm.perprocess(df),pr) +"<a href=adddata>"+"<br>"+"<br>"+"<br>"+"<br>"+"<input type=button value='go to predict'/adddata/'></a>"
                                                                            
@app.route('/adddata')
def add():
	return "<h1>add your data</h1>"+"<form action='/about' method='get' ><TABLE BORDER='0'>\
	<tr><th>Dependents       </th>    \t <td><select name='dependence'><option value='1'>yes</option><option value='0'>No</option></select><br></td></tr>\
	<tr><th>tenure           </th>     \t <td><input name='tenure' placeholder='tenure' type 'text'><br></td></tr>\
	<tr><th>OnlineSecurity   </th>     \t <td><select name='OnlineSecurity'><option value='1'>yes</option><option value='0'>No</option><option value='2'>No internet service</option></select><br></td></tr>\
	<tr><th>TechSupport      </th>     \t <td><select name='TechSupport'><option value='1'>yes</option><option value='0'>No</option><option value='2'>No internet service </option></select><br></td></tr>\
	<tr><th>Contract         </th>     \t<td><select name='Contract'><option value='0'>month to month</option><option value='1'>one year</option><option value='2'>two year</option></select> <br></td></tr>\
	<tr><th>PaperlessBilling </th>     \t <td><select name='PaperlessBilling'><option value='1'>yes</option><option value='0'>No</option></select><br></td></tr>\
	<tr><th>MonthlyCharges   </th>     \t <td><input name='MonthlyCharges' placeholder='MonthlyCharges' type 'text'><br></td></tr>\
	<tr><th>TotalCharges     </th>     \t <td><input name='TotalCharges' placeholder='TotalCharges' type 'text'><br> </td></tr>\
	<tr><th>select the algorithm </th><td><select name='pr'>\
	<option value='1'>Logistic Regression</option>\
	<option value='2'>K N N</option>\
	<option value='3'>Naive Bayes</option>\
	<option value='4'>S V M</option>\
	<option value='5'>Decision Tree</option>\
	<option value='6'>Random Forest</option></select></td></tr></TABLE>\
	<br><input type='submit' value='predict' >"
@app.route('/about')
def about():
	val=int(request.args["pr"])
	Dependents      =int(request.args["dependence"])
	tenure          =float(request.args["tenure"] )
	OnlineSecurity  =int(request.args["OnlineSecurity"])
	TechSupport     =int(request.args["TechSupport"] )
	Contract	    =int(request.args["Contract"] )
	PaperlessBilling=int(request.args["PaperlessBilling"])
	MonthlyCharges  =float(request.args["MonthlyCharges"])
	TotalCharges    =float(request.args["TotalCharges"])
	#data0=([Dependents,tenure,OnlineSecurity,TechSupport,Contract,PaperlessBilling,MonthlyCharges,TotalCharges])
	data0 =({'Dependents':[Dependents],'tenure':[tenure],'OnlineSecurity':[OnlineSecurity],'TechSupport':[TechSupport],'Contract':[Contract],'PaperlessBilling':[PaperlessBilling],'MonthlyCharges':[MonthlyCharges],'TotalCharges':[TotalCharges]})
	data=pd.DataFrame(data0)
	return str(algorithm.predData(algorithm.perprocess(df),data,val)) 
	
	
	
	


if __name__ == '__main__':
    app.run(debug=True,host="127.0.0.1",port=5000)
