import streamlit as st
import model
st.markdown('<style>body{text-align: center;}</style>', unsafe_allow_html=True)


# About section (sidebar)
st.sidebar.image('rentalimg.jpeg', width=200)
st.sidebar.subheader('About')
st.sidebar.info('Agri-Brilliance is a web application which can help you classify plant and crop diseases using machine learning.')

# Instructions section (sidebar)
st.sidebar.subheader('Instructions to use')
st.sidebar.info("Using the app is very simple. All you have to do is upload an image of the diseased plant's (or crop's) leaf and click on the __Predict__ button. \
The app will use machine learning to predict the disease and display the result along with a probability percentage.")

# Main app interface
st.title('Agri-Brilliance')
st.header('A plant and crop disease detection app')
st.text('')

img = st.file_uploader(label='Upload leaf image (PNG, JPG or JPEG)', type=['png', 'jpg', 'jpeg'])
if img is not None:
    predict_button = st.button(label='Predict')
    if predict_button:
        st.text('')
        st.text('')
        st.image(image=img.read(), caption='Uploaded image')
        prediction_class, prediction_probability = model.predict(img)
        st.subheader('Prediction')
        st.info(f'Classification: {prediction_class}, Probability: {prediction_probability}%')

st.markdown("***")
# Information section (sidebar)
st.header('More information')
st.info('As of now, the app can detect the following 4 main classes - Corn ,Grapes ,Potato ,Tomato\
    We will be adding more disease classes to the app soon\
    We have also planned to add possible remedies of these diseases to the app.')

st.markdown("***")
st.header('Model Summary')
st.image('summary.png', width=500)
