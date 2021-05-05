import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("decision_model.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('CLASSIFICATION DATASET.csv')
# Extracting independent variable:
X = dataset.iloc[:,0:14].values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True) 
imputer = imputer.fit(X[:, 1:5])  
X[:, 1:5]= imputer.transform(X[:, 1:5])  
imputer = imputer.fit(X[:, 7:13])  
X[:, 7:13]= imputer.transform(X[:, 7:13])

imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value="female", verbose=1, copy=True)
imputer = imputer.fit(X[:, 5:6])   
X[:, 5:6]= imputer.transform(X[:, 5:6])
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'constant', fill_value="", verbose=1, copy=True)
imputer = imputer.fit(X[:, 6:7])   
X[:, 6:7]= imputer.transform(X[:, 6:7])

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
labelencoder_X = LabelEncoder()
X[:, 6] = labelencoder_X.fit_transform(X[:, 6])


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(age,cp,trestbps,chol, fbs,
                                Gender, Geography,restecg,
                                thalach, exang,oldpeak, slope,
                                ca, thal):
  output= model.predict(sc.transform([[age,cp,trestbps,chol, fbs,
                                      Gender, Geography,restecg,
                                      thalach, exang,oldpeak, slope,
                                      ca, thal]]))
  print("Heart Disease", output)
  if output==[0]:
    prediction ="It is HeartDisease Catogory 0"
  elif output==[1]:
    prediction ="It is HeartDisease Catogory 1"
  elif output==[2]:
    prediction ="It is HeartDisease Catogory 2"
  elif output==[3]:
    prediction ="It is HeartDisease Catogory 3"
  elif output==[4]:
    prediction ="It is HeartDisease Catogory 4"
  print(prediction)
  return prediction
def main():
    st.title("HeartDisease Prediction")
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Ml Lab Experiment Deployment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    age = st.text_input("age","Type Here")
    cp = st.text_input("cp","Type Here")
    trestbps = st.text_input("trestbps","Type Here")
    chol = st.text_input("chol","Type Here")
    fbs = st.text_input("fbs","Type Here")
    Gender = st.text_input("Gender","Type Here")
    Geography = st.text_input("Geography","Type Here")
    restecg = st.text_input("restecg","Type Here")
    thalach = st.text_input("thalach","Type Here")
    exang = st.text_input("exang","Type Here")
    oldpeak = st.text_input("oldpeak","Type Here")
    slope = st.text_input("slope","Type Here")
    ca = st.text_input("ca","Type Here")
    thal = st.text_input("thal","Type Here")

    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(age,cp,trestbps,chol, fbs,
                                          Gender, Geography,restecg,
                                          thalach, exang,oldpeak, slope,
                                          ca, thal)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.text("Developed by Pritesh Kumar")

if __name__=='__main__':
  main()
   