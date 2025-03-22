from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation, NMF
from topic_extraction import lda_model, nmf_model

count_data = pd.read_csv('count_data.csv')
tfidf_data = pd.read_csv('tfidf_data.csv')

# Reusable function to draw word clouds
def plot_wordclouds(model, feature_names, title_prefix):
    n_topics = model.n_components
    for topic_idx, topic in enumerate(model.components_):
        top_words = {feature_names[i]: topic[i] for i in topic.argsort()[-20:][::-1]}
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_words)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(f"{title_prefix} Topic {topic_idx + 1}", fontsize=14)
        plt.tight_layout()
        plt.show()

# Generate word clouds
plot_wordclouds(lda_model, count_data.columns[1:], title_prefix="LDA")
plot_wordclouds(nmf_model, tfidf_data.columns[1:], title_prefix="NMF")

def plot_top_words_bar(model, feature_names, title_prefix, n_words=10):
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        weights = topic[top_indices]

        plt.figure(figsize=(8, 4))
        sns.barplot(x=weights, y=top_words)
        plt.title(f"{title_prefix} Topic {topic_idx + 1} - Top {n_words} Words")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()



#Topic count for LDA and NMF
data = pd.read_csv('data_with_topics.csv')
data['LDA_Topic_Label'].value_counts().plot(kind='bar', title='LDA Topic Distribution')
plt.xlabel('Topic name')
plt.ylabel('Count')
plt.show()


data['NMF_Topic_Label'].value_counts().plot(kind='bar', title='NMF Topic Distribution')
plt.xlabel('Topic Name')
plt.ylabel('Count')
plt.show()


