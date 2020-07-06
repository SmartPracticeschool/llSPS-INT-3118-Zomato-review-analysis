from tensorflow.keras.models import load_model
import numpy as np
import pickle
from  sklearn.feature_extraction.text import CountVectorizer
model=load_model('model.h5')
loaded=CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))
da="The food was excellent and service was very good."
da= da.split("delimiter")
result=model.predict(loaded.transform(da))
print(result)
prediction=result>0.5
print(prediction)
