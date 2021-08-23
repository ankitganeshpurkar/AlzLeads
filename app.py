# Import libraries
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


st.write(""" # AlzLeads : A Web Application for Prediction of Inhibitor for Alzheimer's Disease   """)

user_input = st.text_input("Enter the SMILES string", ' ')
try:
    mol = Chem.MolFromSmiles(user_input)
except:
    st.write("Enter the correct smiles string.")
df_des =  pd.DataFrame()
calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
header = list(calc.GetDescriptorNames())
d2 = list(calc.CalcDescriptors(mol))
d = {x:y for x,y in zip(header, d2)}
df_des =  df_des.append(d, ignore_index = True)

result = [-1]
target_name = st.selectbox('Select target', ('AChE', 'BChE', 'BACE1', 'GSK3B', 'MAOB', 'N2B'))

if user_input == " ":
    result[0] = -1
    
else:
    if target_name == "AChE":
        feat = pickle.load(open('.\data\Feature_AChE.pkl', 'rb'))
        X = df_des[feat]
        option1 = st.selectbox('Select the model', ('Random forest', 'SVC'))
        if option1 == 'Random forest':
            loaded_model = pickle.load(open('.\data\Final_model_AChE_RF.pkl', 'rb'))
            result = loaded_model.predict(X)
                
        elif option1 == 'SVC':
            loaded_model = pickle.load(open('.\data\Final_model_AChE_SVC.pkl', 'rb'))
            scaler = pickle.load(open('.\data\ss_AChE.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)

    elif target_name == "BChE":
        feat = pickle.load(open('.\data\Feature_BChE.pkl', 'rb'))
        X = df_des[feat]
        option1 = st.selectbox('Select the model', ('Random forest',))
        if option1 == 'Random forest':
            loaded_model = pickle.load(open('.\data\Final_model_BChE_RF.pkl', 'rb'))
            result = loaded_model.predict(X)
                
    elif target_name == "BACE1":
        feat = pickle.load(open('.\data\Feature_BACE1.pkl', 'rb'))
        X = df_des[feat]
        option1 = st.selectbox('Select the model', ('Random forest', 'SVC', 'KNN'))
        if option1 == 'Random forest':
            loaded_model = pickle.load(open('.\data\Final_model_BACE1_RF.pkl', 'rb'))
            result = loaded_model.predict(X)
                
        elif option1 == 'SVC':
            loaded_model = pickle.load(open('.\data\Final_model_BACE1_SVC.pkl', 'rb'))
            scaler = pickle.load(open('.\data\ss_BACE1.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)
            
        elif option1 == 'KNN':
            loaded_model = pickle.load(open('.\data\Final_model_BACE1_KNN.pkl', 'rb'))
            scaler = pickle.load(open('.\data\ss_BACE1.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)

    if target_name == "GSK3B":
        feat = pickle.load(open('.\data\Feature_GSK3B.pkl', 'rb'))
        X = df_des[feat]
        option1 = st.selectbox('Select the model', ('Random forest', 'SVC', 'KNN'))
        if option1 == 'Random forest':
            loaded_model = pickle.load(open('.\data\Final_model_GSK3B_RF.pkl', 'rb'))
            result = loaded_model.predict(X)
                
        elif option1 == 'SVC':
            loaded_model = pickle.load(open('.\data\Final_model_GSK3B_SVC.pkl', 'rb'))
            scaler = pickle.load(open('.\data\ss_GSK3B.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)
            
        elif option1 == 'KNN':
            loaded_model = pickle.load(open('.\data\Final_model_GSK3B_KNN.pkl', 'rb'))
            scaler = pickle.load(open('.\data\ss_GSK3B.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)

    elif target_name == "MAOB":
        feat = pickle.load(open('.\data\Feature_MAOB.pkl', 'rb'))
        X = df_des[feat]
        option1 = st.selectbox('Select the model', ('Random forest',))
        if option1 == 'Random forest':
            loaded_model = pickle.load(open('.\data\Final_model_MAOB_RF.pkl', 'rb'))
            result = loaded_model.predict(X)
                
    elif target_name == "N2B":
        feat = pickle.load(open('.\data\Feature_N2B.pkl', 'rb'))
        X = df_des[feat]
        option1 = st.selectbox('Select the model', ('Random forest', 'Decision tree', 'SVC', 'KNN'))
        if option1 == 'Random forest':
            loaded_model = pickle.load(open('.\data\Final_model_N2B_RF.pkl', 'rb'))
            result = loaded_model.predict(X)
            
        elif option1 == 'Decision tree':
            loaded_model = pickle.load(open('.\data\Final_model_N2B_DT.pkl', 'rb'))
            result = loaded_model.predict(X)
                
        elif option1 == 'SVC':
            loaded_model = pickle.load(open('.\data\Final_model_N2B_SVC.pkl', 'rb'))
            scaler = pickle.load(open('.\data\ss_N2B.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)
            
        elif option1 == 'KNN':
            loaded_model = pickle.load(open('.\data\Final_model_N2B_KNN.pkl', 'rb'))
            scaler = pickle.load(open('.\data\ss_N2B.pkl', 'rb'))
            Xtest = scaler.transform(X)
            result = loaded_model.predict(Xtest)
     
st.header('Prediction')
if result[0] == 1:
    st.write("Active")
elif result[0] == 0:
    st.write("Moderately Active")
elif result[0] == -1:
    st.write("Enter the correct smiles string.")
    
    
st.write("""

Key: IC50 <= 5000 nM : Active and IC50 > 5000 nM : Moderately Active  """)
