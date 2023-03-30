import json

import streamlit as st
import requests
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import pandas as pd

from PIL import Image
from utiles import get_crypto_data, get_x_y, price_prediction, plot_last_month


st.set_page_config(initial_sidebar_state="collapsed")

tab1, tab2, tab3 = st.tabs(["💸 Home", "📈 Prediction", "⚽️ The team"])

with tab1:
    with open("designing.css") as source_des :
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>CPP<br>- Crypto Price Prediction -<br> Or how to get rich</h1>", unsafe_allow_html=True)

    st.markdown("---")


    st.markdown('''

        Welcome to our MVP.\n
        Here you will find our bitcoin price evolution prediction!
        It will give you the expected price in 24h.

        ''')

    st.markdown("Beware, this is a prediction, not a certainty!")



    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_json = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_27klftwl.json")

    st_lottie(lottie_json, height=300)

    st.markdown('''
        _This prediction is provided by : Alexis V. A., Amaury G., Aran K. & Vincent M._
        ''')

with tab2 :

    url = 'https://crypto-j2nsa5srea-ew.a.run.app/predict'


    response = requests.get('https://crypto-j2nsa5srea-ew.a.run.app/predict')
    json_data = json.loads(response.text)


    lottie_pred = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_a6jfvjhr.json")
    st_lottie(lottie_pred, height=200)


    if st.button("Click to get our estimation of the bitcoin price in 24h"):


        df = get_crypto_data()
        fig, ax = plot_last_month(df, json_data["result"])
        st.pyplot(fig)
        #st.text(response.json())
        #def load_lottieurl(url: str):
           # r = requests.get(url)
            #if r.status_code != 200:
            #    return None
           # return json_data(2)
        #col1, col2 = st.columns(2)

        # Add content to the left column
        #with col1:
            #df = get_crypto_data()
            #st.pyplot(plot_last_month(df, json_data["result"]))

        #st.set_option('deprecation.showPyplotGlobalUse', False)

        #st.pyplot(plot_last_month(df, json_data["result"]), clear_figure=False)


        # Adjust the width of the plot to align to the left
        #plot._repr_html_ = plot._repr_html_.replace("<div", "<div style='margin-left: 0; margin-right: auto;'")




with tab3:

    st.markdown("""
            Our team of four worked on the CPP project. You will find a brief description. But be sure that it is by speaking abour the club they support that they created a climat of trust and workable environment
            """)

    st.markdown("---")

    with open("designing.css") as source_des :
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

    image1 = Image.open('logo/logo_real.jpeg')

    st.image(image1, width=130,)

    st.markdown(" **Alexis :** He initiated the project and was responsible of scapping the data from the API. As he describe himself, 21, Brussels, Chess, Crypto, Combat Sports, Finance and Engineering. Planning on freelancing as a Data Scientist as well as creating apps. Tempus Fugit ")

    image2 = Image.open('logo/logo_psg.jpeg')

    st.image(image2, width=150,)

    st.markdown("**Amaury :** He wisely joinded the project and was in charge of the backend. He works as a Data Analyst at Wave.ai, a small business that aims at revolutionizing the way professional coaching works (find out more [here](https://www.wave.ai/)). He lives in Saint-Germain-en-Laye near Paris, France, with his wife and dog Jackson.")

    image3 = Image.open('logo/logo_arsenal.jpeg')

    st.image(image3, width=150,)

    st.markdown("**Aran :** Lucky us, he jumped in the project to take care of the model. He is a computer science student and plan to go into Artificial Intelligence, using his behind the table skills to make better decisions with his algorithms")

    image4 = Image.open('logo/logo_united.jpeg')

    st.image(image4, width=150,)

    st.markdown("**Vincent :** As the man in charge of the frontend, he chose to put the biggest club in the world at the end of this page. He is a project manager in traceability (his [compagny](https://www.productdna.com/en/)). He aim to learn, develop and use new skills to built a better and smarter humanised world !")
