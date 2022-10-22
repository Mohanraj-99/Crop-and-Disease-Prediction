from __future__ import division, unicode_literals

# Import libraries.
# pip install pandas==1.1.0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score
from sklearn.decomposition import TruncatedSVD
import itertools

import matplotlib.pyplot as plt
from matplotlib import rc

import altair as alt

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)
enc = LabelEncoder()

def run():
    st.sidebar.image('rentalimg.jpeg', width=200)
    dataset_name = st.sidebar.selectbox("Choose.", ("Crop Prediction", "Soil Testing Centers"))

    # LOAD DATA.
    if dataset_name == "Crop Prediction":
        dataset_name = "data soil"+".csv"
        try:
            data = pd.read_csv(dataset_name)
        except:
            st.error("No such file could be found in the working directory. Make sure it is there and it is a csv-file.")
        st.write("""
            # Crop Prediction App
            This application predicts the **Crop** based on the soil characteristics !
            """)
        # DEFINE DATA.
        st.header("Preprocessing")
        st.write("First, data has to be preprocessed by adjusting the number of classes before transforming the data.",
        "As you can imagine, not every **class (specialty)** might be represented properly in a data set to classify it adequately.",
        "It is advisable to remove classes which are underrepresented and whose abundance is below some treshold to achieve a better classification performance.",
        "(Alternatively, try to collect more data.)")
        st.markdown("***")
        st.header("Load data")
        st.write("Display sample data")
        df1 = data.sample(n=10)
        df1.reset_index(drop=True, inplace=True)
        st.dataframe(df1)
        labels = enc.fit_transform(data['crop'])
        labels = np.ravel(labels)
        unique_values, counts = np.unique(labels, return_counts=True)
        relative_counts = counts/np.sum(counts)
        label_names = enc.inverse_transform(unique_values)
        st.write("The data set contains", np.shape(unique_values)[0], "classes and", data.shape[0], "samples.")

        # INSTRUCTION
        rel_counts = pd.DataFrame(data=relative_counts, columns=["fraction of class in the data set"]).set_index(label_names)
        st.write(rel_counts)

        # DATA TRANSFORMATION
        data['Region']= enc.fit_transform(data['Region'])
        train, test = train_test_split(data, test_size = 0.2, stratify = data['Region'], random_state = 42)
        X_train = train[['Region','pH','N','P','K','temperature','humidity','rainfall']] # taking the training data features
        y_train = train.crop # output of the training data
        X_test = test[['Region','pH','N','P','K','temperature','humidity','rainfall']] # taking test data feature
        y_test = test.crop # output value of the test data
        
        def user_input_features():
            st.sidebar.header("Specify Input Parameters")
            pH = st.sidebar.slider('pH of the soil', 6.0, 9.0, 7.4)
            N = st.sidebar.slider('Nitrogen', 80, 190, 120)
            P = st.sidebar.slider('Phosphorus', 30, 120, 70)
            K = st.sidebar.slider('Potassium', 40, 300, 60)
            Area = st.sidebar.selectbox('Area of Kancheepuram',('0-Aathanancherry', '1-Chirukalathur', '2-Erumaiyur', '3-Manimangalam', 
            '4-Mutalivakkam','5-Naduveerapattu', '6-Nanthampakkam', '7-Natarasanpattu', '8-Neelamangalam', '9-Oorathur',
            '10-Padapai', '11-Palantantalam', '12-Periyapanicherry', '13-Poodi', '14-Puntantalam', '15-Saalamangalam',
            '16-Somangalam', '17-Umaiyalparancherry', '18-Vadakupattu'))
            data = {'Region': int(Area.split('-')[0]),
            'pH': pH,
            'N': N,
            'P': P,
            'K': K,
            'temperature' : 27.4,
            'humidity' : 62,
            'rainfall' : 111
            }
            features = pd.DataFrame(data, index=[0])
            return features

        df = user_input_features()

        st.markdown("***")
        st.header("Data Input parameters")
        st.info("The default data is passed initially for processing.")
        st.write("Using the sidebars you can provide the manual inputs too")
        st.subheader("Specified input parameters")
        st.dataframe(df)
        st.sidebar.header('Customizing model.')
        classifier_name = st.sidebar.selectbox('Select classifier',('KNN', 'SVM', 'Random Forest'))

        def add_parameter_ui(clf_name):
            params = dict()
            if clf_name == 'SVM':
                C = st.sidebar.slider('C', 0.01, 10.0)
                params['C'] = C
            elif clf_name == 'KNN':
                K = st.sidebar.slider('K', 1, 15)
                params['K'] = K
            else:
                max_depth = st.sidebar.slider('max_depth', 2, 15)
                params['max_depth'] = max_depth
                n_estimators = st.sidebar.slider('n_estimators', 1, 100)
                params['n_estimators'] = n_estimators
            features = pd.DataFrame(params, index=[0])
            return params,features
        params,features = add_parameter_ui(classifier_name)

        def get_classifier(clf_name, params):
            clf = None
            if clf_name == 'SVM':
                clf = SVC(C=params['C'])
            elif clf_name == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=params['K'])
            else:
                clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], random_state=42)
            return clf
        st.markdown("***")
        st.header("Training")
        st.info("The model is based on a **KNN** by default with its default hyperparameters")
        st.write("You can customize the with others models and its hyperparameters in the sidebar")
        st.subheader(f"The hyperparameter for the {classifier_name} algorithm")
        st.dataframe(features)

        # MODEL BUILDING.
        def train_model(classifier_name, params,X_train, y_train):
            model = get_classifier(classifier_name, params)
            model.fit(X_train, y_train)
            return model

        clf = train_model(classifier_name, params,X_train, y_train)

        st.markdown("***")
        
        prediction = clf.predict(df)
        
        st.header("The Predicted Crop is ")
        st.write(prediction)
        st.markdown("***")

        # MODEL EVALUATION
        st.header("Evaluation")
        y_pred = clf.predict(X_test)
        f1_score_ = f1_score(y_test, y_pred, average="weighted")
        st.write("The **F1 score** is",np.round(f1_score_, 2), ".")
        st.write("Below, the **confusion matrix** provided.")
        cm = confusion_matrix(y_test, y_pred)
        cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
        labels_repeated = []
        for _ in range(np.unique(label_names).shape[0]):
            labels_repeated.append(np.unique(label_names))
        source = pd.DataFrame({'predicted class': np.transpose(np.array(labels_repeated)).ravel(),
                            'true class': np.array(labels_repeated).ravel(),
                            'fraction': np.round(cm.ravel(), 2)})
        heat = alt.Chart(source, height=400 , width=400,title="confusion matrix").mark_rect(opacity=0.7).encode(
            x='predicted class:N',
            y='true class:N',
            color=alt.Color('fraction:Q', scale=alt.Scale(scheme='blues')),
            tooltip="fraction")
        st.altair_chart(heat)
        st.write("The Accuracy on the test data ",  accuracy_score(y_pred, y_test), ".")
    
    else:
        location = st.sidebar.selectbox("Location",("Kanchipuram",'Chennai'))
        try:
            data = pd.read_csv("agricultureoffice.csv")
        except:
            st.error("No such file could be found in the working directory. Make sure it is there and it is a csv-file.")
        st.write("""
            # Soil Testing Centers
            The Centers for testing the soil characteristics within Chennai and Kanchipuram.
            """)
        st.info("By default the location is **Kanchipuram** ")
        st.write("You can use the Location dropdown for other districts")
        data.reset_index(drop=True, inplace=True)
        def display(data,location):
            df = data[data.ReferenceLocation == location]
            df.reset_index(drop=True, inplace=True)
            return df
        st.table(display(data,location))
    
if __name__ == "__main__":
    run()