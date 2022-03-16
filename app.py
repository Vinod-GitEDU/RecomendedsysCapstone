from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import warnings
import model
from model import recommender
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
# import xgboost


app = Flask(__name__)

# with open('Model/lr_model.pkl','rb') as fp:
# 	model = pickle.load(fp)
		
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	Exp = request.form.get('user_input')
	Input = Exp
	prediction = model.recommender(Input)
	if type(prediction) == str:
		return render_template('index.html', OUTPUT="No such user exists")
	else:
		return render_template('index.html', user_val=Input,OUTPUT=prediction.to_html())


if __name__ == "__main__":
    app.run(debug=True)