import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
#image = Image.open('Salesforce.jpg')

#st.image(image, width=150)
st.write("""
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/mevijaygupta2010/Data-Science/mevijaygupta2010-ML/penguins-model/penguins_pred.csv?token=AJOL2WPBJYUG6OD7442V3ODAP2K3S)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    #st.write(input_df)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
penguins_raw = pd.read_csv('Penguins_Cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)
no_of_row=input_df.shape[0]
#st.write(df)
# Encoding of ordinal features
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:no_of_row] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
penguins_species = np.array(['Adelie','Chinstrap','Gentoo'])
st.write(penguins_species[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)

#st.write(prediction_proba[0,0])

result_df=pd.DataFrame(penguins_species[prediction],columns=['species'])#pd.concat([df],axis=1)
def return_prob(i):
    if penguins_species[prediction] == 'Adelie':
        return prediction_proba[i,0]
    elif penguins_species[prediction] == 'Chinstrap':
        return prediction_proba[i,1]
    else:
        return prediction_proba[i,2]

#st.write(return_prob(0))

result_df_1=pd.concat([input_df,result_df],axis=1)

#for j in range(no_of_row):
#    st.write(return_prob(j))

    #result_df_1['Prediction_Probabilty']=return_prob(j)*100
st.subheader('Results')
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="Penguins_prediction.csv">Download CSV File</a>'
    return href
st.write(result_df_1)
st.markdown(filedownload(result_df_1), unsafe_allow_html=True)
