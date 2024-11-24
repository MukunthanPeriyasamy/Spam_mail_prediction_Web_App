import pandas as pd
import streamlit as st
import numpy as np

# """Loading Dataset"""

dataset = pd.read_csv('./datasets/mail_data.csv')

# """#Replacing Null values with Empty Null String"""

dataset = dataset.where(pd.notnull(dataset),'')

# """Label "Spam" as 0 and "Ham" as 1"""
dataset.loc[dataset['Category']== 'spam','Category'] = 0
dataset.loc[dataset['Category']== 'ham','Category'] = 1
dataset.head()

# """Feature and Label Data"""
y = dataset.iloc[:,:-1].values
x = dataset.iloc[:,-1].values

# """Splitting Training and Testing Data"""

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# """Transform Text Into Feature Vector"""

from sklearn.feature_extraction.text import TfidfVectorizer
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english')

x_train_feature = feature_extraction.fit_transform(x_train)
x_test_feature = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# """Training Logistic Model"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train_feature,y_train.ravel())

# """Building a Predictive System"""

def main():

  st.title('Spam Mail Prediction 📧🚀')
  mail = st.text_input("Enter the mail")
  input_mail = [mail]
  input_feature_mail = feature_extraction.transform(input_mail)
  prediction = classifier.predict(input_feature_mail)

  if prediction == 1:
    res = "Ham/Non-Spam Mail"
  else:
    res = "Spam Mail"

  if st.button("Check"):
    st.write(res)

if __name__ == "__main__":
  main()