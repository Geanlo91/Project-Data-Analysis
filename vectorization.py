from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('preprocessed_data.csv')

# Vectorizing the data
count_vectorizer = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

count_matrix = count_vectorizer.fit_transform(data['Reports'])
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Reports'])

#convert to dataframe for comparison
count_df = pd.DataFrame(count_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())


#save the vectorized data
count_df.to_csv('count_data.csv', index=False)
tfidf_df.to_csv('tfidf_data.csv', index=False)


 #comparing the two vectorization methods
 #Top 20 words by total frequency in BOW and total weight in TF-IDF
bow_top_words = count_df.sum().sort_values(ascending=False).head(20)
tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(20)

#plotting bar chart side by side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
bow_top_words.plot(kind='barh', color='blue')
plt.title('Top 20 Words by Count Vectorization')
plt.gca().invert_yaxis()

plt.subplot(1, 2, 2)
tfidf_top_words.plot(kind='barh', color='green')
plt.title('Top 20 Words by TF-IDF Vectorization')
plt.gca().invert_yaxis()
plt.tight_layout()

plt.tight_layout()
plt.show()



