#import Packages:
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Human Resource Information System")
st.write("A human resource information system (HRIS) is software that provides a centralized repository of employee master data that the human resource management (HRM) group needs for completing core human resource (core HR) processes. An HRIS can help HR and organizations become more efficient through the use of technology.")
st.title("HR Reporting")
st.write("Human Resource report/HR Report is a way to showcase and analyze all your human resources related metrics, stats, and numbers in the best way possible. Human Resources Reports help to identify areas of improvement and understand different HR functions like employee performance, retention, hiring strategies, and so on.")
st.title("HR Accounting")
st.write("Human resource accounting involves the tracking of all costs related to employees in a separate report. These costs include employee compensation, payroll taxes, benefits, training, and recruiting.")
st.title("HR Evaluation")
st.write("Evaluations are a mechanism to provide feedback and documentation about an employee's performance through a defined time period, and can provide clear communication of job expectations and goals.")
df=pd.read_csv("C:/Users/VARATHAMOHAN/StreamLit_Files/CAT_03/employee_promotion.csv")
df=df.dropna()
st.sidebar.title("Contents")
if st.sidebar.button("Features"):
    st.title("Features")
    #Unique values in the data
    col=['department','region','recruitment_channel','education']

    for i in col:
        st.title(i)
        st.write(df[i].unique())
        st.write(df[i].nunique())
        st.write()
#Using LabelEncoder To transform the data
from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in df.columns:
    if df[i].dtype == 'object':
        df[i]=l.fit_transform(df[i])
        
# The 'object' data type columns are transformed as follows

x=df.drop('is_promoted',axis=1)
y=df['is_promoted']
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.30)
from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler(feature_range=(0,1))
xtrain=mms.fit_transform(xtrain)
xtest=mms.fit_transform(xtest)
xtrain=pd.DataFrame(xtrain)
xtest=pd.DataFrame(xtest)
if st.sidebar.button("Visualization"):
    df=pd.read_csv("C:/Users/VARATHAMOHAN/StreamLit_Files/CAT_03/employee_promotion.csv")
    #Number of employess and their education status - Promotion
    st.title("Employees and their Education status")
    plt.figure(figsize=(6,6))
    sns.barplot(x='education',y='employee_id',hue='is_promoted',data=df)
    plt.show()
    st.pyplot()
    st.title("Inference:")
    st.write("By using barplot,we can see number of employees and their education status.The x-axis indicates education and the y-axis indicates employees")
    st.title("Education status and Awards won")
    fig4=px.pie(df,values='awards_won',names='education',template='plotly_dark')
    st.plotly_chart(fig4,use_container_width=True)  
    st.title("Inference:")
    st.write("The employees who has bachelor as their education status holds 70% of the awards and 857 employees holds the awards")
    st.write("The employees who has masters as their education status holds 28% of the awards and 343 employees holds the awards")
    st.write("The employees who has below secondary as their education status holds 1.15% of the awards and 14 employees holds the awards")
    st.title("Employees Count")
    sns.countplot(x='gender',data=df,palette='Greens_d')
    st.pyplot()
    st.title("Inference:")
    st.write("There are approximately 15000 female employees and 35000 male employees")
    st.title("Department")
    fig=px.violin(df,df['department'],template='plotly_dark')
    st.plotly_chart(fig,use_container_width=True)
    st.title("Service varies with age among males and females ")
    fig1=px.box(df,df['age'],df['length_of_service'],color='gender',title='Length of Service',template='plotly_dark')
    st.plotly_chart(fig1,use_container_width=True)
    st.title("Department wise promotion")
    fig2=px.pie(df,values='is_promoted',names='department',template = 'plotly_dark')
    st.plotly_chart(fig2,use_container_width=True)
    st.title("Inference:")
    st.write("25% of the employees from sales and marketing has been promoted")
    st.write("22% of the employees from operations has been promoted")
    st.write("16.7% of the employees from Technology has been promoted")
    st.write("15% of the employees from Procurement has been promoted")
    st.write("10% of the employees from Analytics has been promoted")
    st.write("4% of the employees from Finance has been promoted")
    st.write("2% of the employees from HR department  has been promoted")             
    
if st.sidebar.button("Analyze"):
    #Logistic, Decision tree, random forest
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    lr=LogisticRegression()
    dc=DecisionTreeClassifier()
    rf=RandomForestClassifier()

    model=[lr,dc,rf]

    for models in model:
        models.fit(xtrain,ytrain)

        ypred=models.predict(xtest)


        st.text('Model :')
        st.write(models)
        st.text('-----------------------------------------------------------------------------------------------------------------------')
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score

        st.write('confusion matrix :')
        st.write(confusion_matrix(ytest,ypred))
        st.write('classification report:')
        st.write(classification_report(ytest,ypred))
        st.write('accuracy             :')
        st.write(round(accuracy_score(ytest,ypred),2))
        st.write('precision            :')
        st.write(round(precision_score(ytest,ypred),2))
        st.write('recall               :')
        st.write(round(recall_score(ytest,ypred),2))
        st.write('f1                   :')
        st.write(round(f1_score(ytest,ypred),2))
        

       

