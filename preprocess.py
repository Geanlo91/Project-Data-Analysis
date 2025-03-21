import pandas as pd
import numpy as np
import os
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

data = pd.read_csv('Datasetprojpowerbi.csv')


def preprocess(data):
    # Dropping all columns except Reports
          data = data.drop(['Genre', 'Age','Gpa','Year','Count','Gender','Nationality'],axis=1)
          # Removing all special characters
          data['Reports'] = data['Reports'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x))
          #converting to lower case
          data['Reports'] = data['Reports'].apply(lambda x: x.lower())

          # Removing all stopwords
          sw = set(stopwords.words('english'))
          data['Reports'] = data['Reports'].apply(lambda x: ' '.join([word for word in x.split() if word not in sw]))

          #Removeing all numbers
          data['Reports'] = data['Reports'].apply(lambda x: re.sub(r'\d+', '', x))

          #Removing all extra spaces
          data['Reports'] = data['Reports'].apply(lambda x: re.sub(r'\s+', ' ', x))

          #Applying Lemmatization
          lemmatizer = WordNetLemmatizer()
          data['Reports'] = data['Reports'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

          return data
try:
    data = preprocess(data)
    
except Exception as e:
    print(f"An error occured during preprocessing: {e}")      

# Saving the preprocessed data
data.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed data saved to preprocessed_data.csv")
print(data.head())




