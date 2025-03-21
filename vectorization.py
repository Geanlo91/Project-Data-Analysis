from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

data = pd.read_csv('preprocessed_data.csv')

# Vectorizing the data
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

count_matrix = count_vectorizer.fit_transform(data['Reports'])
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Reports'])

#convert to dataframe for comparison
count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

#print the dataframes
print("count vectorizer representation:\n", count_df.head())
print("tfidf vectorizer representation:\n", tfidf_df.head())
print("same feautures:",set(count_df.columns) == set(tfidf_df.columns))

#save the vectorized data
count_df.to_csv('count_data.csv', index=False)
tfidf_df.to_csv('tfidf_data.csv', index=False)


