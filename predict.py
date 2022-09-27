import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

logreg = data["model"]
le_mobility = data["le_mobility"]


def show_predict_page():
    st.title("Patient participation in a healthcare program prediction")
    
    st.write("""### We need some information to predict participation""")
    
    mobility = (
         "Car",
        "No car",
    )
    
   
    
    mobility = st.selectbox("Do you have a Car?",mobility)
    
    age = st.slider("Your age",20,90,20)
    
    distance = st.slider("Distance from home to healthcare center",1,100,1)
    
    ok = st.button("Predict")
    if ok:
        X = np.array([[mobility, age, distance ]])
        X[:, 0] = le_mobility.transform(X[:,0])
       
        X = X.astype(float)
        
        participation = logreg.predict(X)
        st.subheader(f"The participation chances are {participation[0]:.2f}")
        st.write(""" Participation : 0 means LOW CHANCES OF PARTICIPATION""")
        st.write(""" Participation : 1 means HIGH CHANCES OF PARTICIPATION""")