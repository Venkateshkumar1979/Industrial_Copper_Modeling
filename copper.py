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



# load the saved model for predicting status using pickle

status = pickle.load(open(r"C:/Users/Venkatesh/copper/classmodel.pkl",'rb'))


status_options = ['Won', 'Draft', 'To be approved', 'Lost', 'Not lost for AM', 'Wonderful', 'Revised', 'Offered', 'Offerable']
item_type_options = ['W', 'WI', 'S', 'Others', 'PL', 'IPL', 'SLAWR']
country_options = ['28.0', '25.0', '30.0', '32.0', '38.0', '78.0', '27.0', '77.0','113.0',
                   '79.0', '26.0', '39.0', '40.0', '84.0', '80.0', '107.0','89.0']
application_options = ['10.0', '41.0', '28.0', '59.0', '15.0', '4.0', '38.0', '56.0',
                       '42.0', '26.0', '27.0', '19.0', '20.0', '66.0', '29.0', '22.0',
                       '40.0', '25.0', '67.0', '79.0', '3.0', '99.0', '2.0', '5.0',
                       '39.0', '69.0', '70.0', '65.0', '58.0', '68.0']
# sidebar

with st.sidebar:
    selection=option_menu('Prediction Model',['Selling Price Prediction','Status Prediction'],default_index=0)
    
# Selling price page
if(selection=='Selling Price Prediction'):
    # Page Title
    st.title('Selling Price Prediction using Decision Tree Regressor Model')
    
   
    
    quantity=st.text_input('Quantity in tons')
    country=st.selectbox("Country", sorted(country_options))	
    status=st.selectbox("Status", sorted(status_options))	
    item_type=st.selectbox("Item Type", item_type_options)
    application=st.selectbox("Application", sorted(application_options))	
    thickness=st.text_input('Thickness')	
    width=st.text_input('Width')
    
    # prediction code
    sp=''
    if st.button('Predict Selling Price'):
        # load the saved model for predicting selling price using pickle

        selling_price=pickle.load(open("C:/Users/Venkatesh/copper/DTRmodel.pkl",'rb'))
        sp_scaler=pickle.load(open("C:/Users/Venkatesh/copper/DTRscaler.pkl",'rb'))
        sp_ohe1=pickle.load(open("C:/Users/Venkatesh/copper/itemohe.pkl",'rb'))
        sp_ohe2=pickle.load(open("C:/Users/Venkatesh/copper/statusohe.pkl",'rb'))
        sp_ohe3=pickle.load(open("C:/Users/Venkatesh/copper/applohe.pkl",'rb'))
        sp_ohe4=pickle.load(open("C:/Users/Venkatesh/copper/countryohe.pkl",'rb'))
        # X[['thickness', 'width','qty_4']].values,X_item_ohe,X_status_ohe,X_appl_ohe,X_country_ohe)

        new_data=np.array([[np.log(float(thickness)), float(width), np.power(float(quantity),1/4),item_type,status,application,country]])
        nd_oheA=sp_ohe1.transform(new_data[:,[3]]).toarray()
        nd_oheB=sp_ohe2.transform(new_data[:,[4]]).toarray()
        nd_oheC=sp_ohe3.transform(new_data[:,[5]]).toarray()
        nd_oheD=sp_ohe4.transform(new_data[:,[6]]).toarray()

        new_data=np.concatenate((new_data[:,[0,1,2,]],nd_oheA,nd_oheB,nd_oheC,nd_oheD),axis=1)
        new_data1=sp_scaler.transform(new_data)
        new_predict=selling_price.predict(new_data1)
        sp=(new_predict**2).round(2)
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
    
    if st.button('Predict Status'):
        status = pickle.load(open(r"C:/Users/Venkatesh/copper/classmodel.pkl",'rb'))

        with open(r'C:/Users/Venkatesh/copper/cscaler.pkl', 'rb') as f:
            cscaler_loaded = pickle.load(f)

        with open(r"C:/Users/Venkatesh/copper/ct.pkl", 'rb') as f:
            ct_loaded = pickle.load(f)
            
        new_samp = np.array([[np.sqrt(float(squantity)),scountry,sapplication,np.log(float(sthickness)),float(swidth),float(ssellingp),sitem_type]])

        new_samp_ohe = ct_loaded.transform(new_samp[:, [6]]).toarray()
        new_samp = np.concatenate((new_samp[:,[0,1,2,3,4,5]], new_samp_ohe), axis=1)
        new_samp = cscaler_loaded.transform(new_samp)
        new_pred = status.predict(new_samp)
        # stat_predict=status.predict([[(float(squantity)**0.5),scountry,sapplication,np.log(float(sthickness)),swidth,sitem_type]])
        # if new_pred[0]==1:
        #     stat='## :blue[The Status is Won !!!]'
        # else:
        #     stat='## :red[The status is Lost!]'
        
        
    st.success(new_pred)

