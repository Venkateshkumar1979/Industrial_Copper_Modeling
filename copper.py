# Importing necessary libraries
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu



st.set_page_config(page_title= "Industrial Copper Modeling",
                     layout= "wide",
                     initial_sidebar_state= "expanded",
                     menu_items={'About': """This ML App is created by Venkatesh Kumar S, GUVI DataScience, Batch-D6162."""})

st.image('Screenshot 2023-07-26 210509.jpg')
st.write('---')

# load the saved model for predicting selling price using pickle

selling_price=pickle.load(open("C:/Users/Venkatesh/copper/sellprice.pkl",'rb'))

# load the saved model for predicting status using pickle

status = pickle.load(open(r"C:/Users/Venkatesh/copper/classmodel.pkl",'rb'))
# cscaler = pickle.load(open(r"C:/Users/Venkatesh/copper/cscaler.pkl",'rb'))
# cohe = pickle.load(open(r"C:/Users/Venkatesh/copper/ct.pkl",'rb'))

#status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
# sidebar

with st.sidebar:
    selection=option_menu('Prediction Model',['Selling Price Prediction','Status Prediction'],default_index=0)
    
# Selling price page
if(selection=='Selling Price Prediction'):
    # Page Title
    st.title('Selling Price Prediction using Decision Tree Regressor Model')
    
   
    
    quantity=st.text_input('Quantity in tons')
    country=st.text_input('Country Code (0-16)')	
    status=st.text_input('Status Code (0-8)')	
    item_type=st.text_input('Item Type code (0-6')	
    application=st.text_input('Application Code (0-29)')	
    thickness=st.text_input('Thickness')	
    width=st.text_input('Width')
    
    # prediction code
    sp=''
    if st.button('Predict Selling Price'):
        sp_predict=selling_price.predict([[(float(quantity)**0.5),country,status,item_type,application,np.log(float(thickness)),width]])
        sp=sp_predict[0].round(2)
    st.success(sp)
        
    

if(selection=='Status Prediction'):
    # Page Title
    st.title('Status Prediction using Decision Tree Classifier Model')
    
    # Taking input
    squantity=st.text_input('Quantity in tons')
    scountry = st.selectbox("Country", sorted(country_options))
    sapplication = st.selectbox("Application", sorted(application_options))
    sthickness=st.text_input('Thickness')	
    swidth=st.text_input('Width')
    ssellingp=st.text_input('Selling Price')
    sitem_type = st.selectbox("Item Type", item_type_options)
    # sitem_type=st.text_input('Item Type code (0-6')	

    
    # prediction code
    stat=''
    if st.button('Predict Status'):
        status = pickle.load(open(r"C:/Users/Venkatesh/copper/classmodel.pkl",'rb'))

        with open(r'C:/Users/Venkatesh/copper/cscaler.pkl', 'rb') as f:
            cscaler_loaded = pickle.load(f)

        with open(r"C:/Users/Venkatesh/copper/ct.pkl", 'rb') as f:
            ct_loaded = pickle.load(f)
            
        new_samp = np.array([[np.sqrt(float(squantity)),scountry,sapplication,np.log(float(sthickness)),swidth,ssellingp,sitem_type]])

        new_samp_ohe = ct_loaded.transform(new_samp[:, [6]]).toarray()
        new_samp = np.concatenate((new_samp[:,[0,1,2,3,4,5]], new_samp_ohe), axis=1)
        new_samp = cscaler_loaded.transform(new_samp)
        new_pred = status.predict(new_samp)
        # stat_predict=status.predict([[(float(squantity)**0.5),scountry,sapplication,np.log(float(sthickness)),swidth,sitem_type]])
        if new_pred[0]==1:
            stat='## :blue[The Status is Won !!!]'
        else:
            stat='## :red[The status is Lost!]'
        
        
    st.success(stat)

