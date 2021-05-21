import pandas as pd 
import streamlit as st 
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('titanic.csv')
df['Male'] = df['Sex'] == 'male'
X = df[['Pclass', 'Male', 'Age', 'Siblings/Spouses', 'Parents/Children', 'Fare']]
y = df['Survived']

model = LogisticRegression()
model.fit(X, y)

st.sidebar.header('User Parameters')

def user_input():
	Pclass = st.sidebar.selectbox('Pclass', [1, 2, 3])
	Sex = st.sidebar.radio('Sex', ['Male(1)', 'Female(0)'])
	if Sex == 'Male(1)':
		Sex = 1
	else:
		Sex = 0
	Age = st.sidebar.slider('Age', 0.4, 80.0, 28.0)
	SibSpo = st.sidebar.slider('Siblings/Spouses', 0, 8, 0)
	ParChi = st.sidebar.slider('Parents/Children', 0, 6, 0)
	Fare = st.sidebar.slider('Fare', 0.0, 513.0, 14.0)
	data = {'Pclass':Pclass,
			'Male':Sex,
			'Age':Age,
			'Siblings/Spouses':SibSpo,
			'Parents/Children':ParChi,
			'Fare':Fare}
	features = pd.DataFrame(data, index=[0])
	return features

st.title('Titanic Passengers Survived Prediction')

st.header('User Parameters')
df = user_input()
st.write(df)

st.header('Prediction Labels')
st.write(pd.DataFrame({'Survived':1, 'Not Survived':0}, index=[0]))

st.header('Prediction')
prediction = model.predict(df)
st.write(prediction)

st.header('Prediction Probability')
prediction_proba = model.predict_proba(df)
st.write(prediction_proba)