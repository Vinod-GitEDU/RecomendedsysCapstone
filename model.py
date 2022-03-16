import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import gc


def recommender(user):
	
	with open('PickleFiles/tfidf_pickle.pkl','rb') as fp:
		tfidf_model = pickle.load(fp)


	with open('PickleFiles/sentimental.pkl','rb') as fp:
		sentimental_model = pickle.load(fp)


	with open('PickleFiles/clean_df.pkl','rb') as fp:
		clean_df_model = pickle.load(fp)


	with open('PickleFiles/Recommender.pkl','rb') as fp:
		user_final_rating_model = pickle.load(fp)

	if len(clean_df_model[clean_df_model.reviews_username==user])>0:
		user_input=user
		recommended_prods_20=user_final_rating_model.loc[user_input].sort_values(ascending=False)[:20]
		products_20_reviews=clean_df_model[clean_df_model.id.isin([x for x in recommended_prods_20.index])][["id","name","final_reviews"]]

		predict_dtm = tfidf_model.transform(products_20_reviews.final_reviews)
		predict_dtm_df=pd.DataFrame(predict_dtm.toarray(), columns=tfidf_model.get_feature_names())

		products_20_reviews["predicted_sentiment"]=sentimental_model.predict(predict_dtm_df)

		products_20_reviews["predicted_sentiment_scorevalue"]=products_20_reviews.predicted_sentiment.apply(lambda x: 1 if x=='Positive' else 0)

		products_20_pivot=products_20_reviews.pivot_table(values="predicted_sentiment_scorevalue",index='name',aggfunc='mean')
		products_20_pivot.sort_values(by="predicted_sentiment_scorevalue",inplace=True,ascending=False)
		final_result= products_20_pivot.head(5)
		return(final_result)
	else:
		return("No such user exists")

	





