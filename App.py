import streamlit as st
import joblib
import requests
import pandas as pd
import numpy as np

import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_option_menu import option_menu

import streamlit.components.v1 as components

st.set_page_config(layout="wide")
with st.sidebar:
    
    choose = option_menu("Welcome", ["Home", "Tech Stack","Analyzer","ML Code", "Contributors"],
                         icons=['house', 'stack', 'robot','code-slash', 'people-fill'],
                         menu_icon="bank", default_index=0, 
                         
                         styles={
                            "container": {"padding": "5!important", "background-color": "#1a1a1a"},
                            "icon": {"color": "White", "font-size": "25px"}, 
                            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#4d4d4d"},
                            "nav-link-selected": {"background-color": "#4d4d4d"},
                        }
    ) 


with open("contributors.html",'r') as f:
   contributors=f.read();
def html():
    components.html(
      contributors
     ,
    height=1400,
    
    scrolling=True,
)
def pred():
    st.title("Whatsapp Chat Analyzer")

    uploaded_file = st.file_uploader("Choose a file")


    if uploaded_file is not None:
        
        bytes_data = uploaded_file.getvalue()
        
        data = bytes_data.decode("utf-8")
        x=data.split('\n')
        #st.write(data)
        y=[]
        for i in x:
            try:
                arr=i.split('-')
                if 'pm' in arr[0] :
                    brr=arr[0].split(',')
                    
                    a=brr[1].split(':')
                    if int(a[0])!=12:
                        a[0]=int(a[0])+12
                    else:
                        a[0]=int(a[0])
                    l=str(brr[0]+', '+str(a[0])+':'+a[1][0:2]+' -'+str(''.join(map(str,arr[1:])))+'\n')
                    y.append(l)
                else:
                    brr=arr[0].split(',')
            
                    a=brr[1].split(':')
                    a[0]=int(a[0])
                    l=str(brr[0]+', '+str(a[0])+':'+a[1][0:2]+' -'+str(''.join(map(str,arr[1:])))+'\n')
                    y.append(l)
            except:
                pass
        t=(''.join(map(str,y)))
        data=t
        
        df = preprocessor.preprocess(data)

        # fetch unique users
        user_list = df['user'].unique().tolist()
        try:
            user_list.remove('group_notification')
        except:
            pass
        user_list.sort()
        user_list.insert(0,"Overall")

        selected_user = st.selectbox("Show analysis wrt",user_list)

        if st.button("Show Analysis"):

            # Stats Area
            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user,df)
            st.title("Top Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.header("Total Messages")
                st.title(num_messages)
            with col2:
                st.header("Total Words")
                st.title(words)
            with col3:
                st.header("Media Shared")
                st.title(num_media_messages)
            with col4:
                st.header("Links Shared")
                st.title(num_links)

            # monthly timeline
            st.title("Monthly Timeline")
            timeline = helper.monthly_timeline(selected_user,df)
            fig,ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'],color='green')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # daily timeline
            st.title("Daily Timeline")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # activity map
            st.title('Activity Map')
            col1,col2 = st.columns(2)

            with col1:
                st.header("Most busy day")
                busy_day = helper.week_activity_map(selected_user,df)
                fig,ax = plt.subplots()
                ax.bar(busy_day.index,busy_day.values,color='purple')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values,color='orange')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.title("Weekly Activity Map")
            user_heatmap = helper.activity_heatmap(selected_user,df)
            fig,ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)

            # finding the busiest users in the group(Group level)
            if selected_user == 'Overall':
                st.title('Most Busy Users')
                x,new_df = helper.most_busy_users(df)
                fig, ax = plt.subplots()

                col1, col2 = st.columns(2)

                with col1:
                    ax.bar(x.index, x.values,color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.dataframe(new_df)

            # WordCloud
            st.title("Wordcloud")
            df_wc = helper.create_wordcloud(selected_user,df)
            fig,ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # most common words
            most_common_df = helper.most_common_words(selected_user,df)

            fig,ax = plt.subplots()

            ax.barh(most_common_df[0],most_common_df[1])
            plt.xticks(rotation='vertical')

            st.title('Most commmon words')
            st.pyplot(fig)



    

with open('techstack.html','r') as f:
  techstack=f.read();
def tech():
    components.html(
    techstack
    ,
    height=1000,
    
    scrolling=True,
    )
def ml():
  st.write("To view the complete code for the end-to-end project, visit our [GitHub](https://github.com/snshahgit/credit-approval-system)")
  components.iframe("https://www.kaggle.com/embed/sns5154/credit-approval-system-val-87-50-test-81-16?kernelSessionId=100312408",height=1000,)





if choose=="Predictor":

    pred()
elif choose=="Home":
    st.title('AI for Healthcare')
    st.markdown("<p style='text-align: justify;'>The objective of the project is to diagnostically predict whether or not a patient has Type 2 diabetes. \nThis predictor is built for Women above 21 years of age. The dataset, originally from the National Institute of Diabetes and Digestive and Kidney Diseases, used for this project consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.</p>", unsafe_allow_html=True)

    # st.markdown("<h1 style='text-align: center;'>Healthcare AI</h1>", unsafe_allow_html=True)

    with open("pic.html",'r') as f:
        pic=f.read();
    components.html(pic, height=400)

    # def load_lottieurl(url: str):
    #     r = requests.get(url)
    #     if r.status_code != 200:
    #         return None
    #     return r.json()
 
    # lt_url_hello = "https://assets6.lottiefiles.com/packages/lf20_1yy002na.json"
    # lottie_hello = load_lottieurl(lt_url_hello)
 
    # st_lottie(
    #         lottie_hello,  
    #         key="hello",
    #         speed=1,
    #         reverse=False,
    #         loop=True,
    #         quality="low",
    #         height=400,
    #         width=400            
    # )

    
elif choose=="Tech Stack":
    tech()
elif choose=="Contributors":
    html()
elif choose=="ML Code":
  ml()
