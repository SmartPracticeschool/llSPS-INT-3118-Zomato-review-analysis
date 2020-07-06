import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
dataset= pd.read_csv('Reviews_zomato.csv')
dataset.drop('name',axis=1,inplace=True)
ds=dataset.iloc[0:,1:3]
ds.isnull().any()
ds['rating_new']=ds['rating'].apply(lambda x:1 if x>=3 else 0)
ds.drop('rating',axis=1,inplace=True)
ds.rename(columns={'rating_new':'rating'},inplace=True)
ds['rating'].value_counts()
ds1=ds.dropna()
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
c=[]
for i in range(0,10000):
    review=re.sub('[^a-zA-Z]',' ',ds1['review'][i])
    review =review.lower()
    review= review.split()
    ps= PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    c.append(review)
from  sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=2000)
X=cv.fit_transform(c).toarray() 
y= ds.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
import pickle
pickle.dump(cv.vocabulary_,open("feature.pkl","wb"))
import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

model=Sequential()
model.add(Dense(input_dim=2000,kernel_initializer="random_uniform",activation="sigmoid",units=1000))
model.add(Dense(kernel_initializer="random_uniform",activation="sigmoid",units=100))
model.add(Dense(units=1,kernel_initializer='random_uniform',activation='sigmoid'))
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
model.fit(x_train,y_train,epochs=50,batch_size=32)
model.save('model.h5')
y_pred=model.predict(x_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

