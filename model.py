import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


#loading the model...
model=joblib.load('house_price_prediction.pkl')


#load the data set...
df=pd.read_csv('clean_house_data.csv')

X=df.drop('price',axis=1)#train data
y=df['price']#target data

#print(X.head())

#spliting the data for training and test....
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)



#feature scalling....
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


#function to predict price....
def predict_house_price(bath,balcony,bhk,total_sqft_int,price_per_sqft,area_type,availability,location):
    #creating a array which can hold the training data columns values...
    x=np.zeros(len(X.columns))
    #print(x)

    x[0]=bath
    x[1]=balcony
    x[2]=bhk
    x[3]=total_sqft_int
    x[4]=price_per_sqft

    if 'area_type_'+area_type in X.columns:
        area_index=np.where(X.columns=='area_type_'+area_type)[0][0]
        #print(area_index)
        x[area_index]=1

    if availability=='Redy To Move':
        x[6]=1

    if 'location_'+location in X.columns:
        loc_index=np.where(X.columns=='location_'+location)[0][0]
        #print(loc_index)
        x[loc_index]=1

    #feature scalling...
    x=sc.transform([x])[0]# taking x as again 1D array...

    #return prediction...
    return round(model.predict([x])[0]*100000,2)


#print(predict_house_price(bath=2,balcony=3,bhk=3,total_sqft_int=1250,price_per_sqft=3520,area_type='Super built-up  Area',availability='n',location=' Devarachikkanahalli'))
