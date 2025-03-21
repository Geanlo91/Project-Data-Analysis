from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the vectorized data
count_data = pd.read_csv('count_data.csv')
tfidf_data = pd.read_csv('tfidf_data.csv')

# Load the original data for reference
data = pd.read_csv('preprocessed_data.csv')

#fitting the LDA model
lda_model = LatentDirichletAllocation(n_components=4, random_state=42)
lda_topics = lda_model.fit_transform(count_data.iloc[:, 1:])  # Exclude the first column (index)

#fitting the NMF model
nmf_model = NMF(n_components=4, random_state=42)
nmf_topics = nmf_model.fit_transform(tfidf_data.iloc[:, 1:])  # Exclude the first column (index)

#View top words in each topic
def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"\nTopic #{topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))


# Display top words for LDA model
display_topics(lda_model, count_data.columns[1:], no_top_words=4)
print("\nLDA Model Topics:\n")

display_topics(nmf_model, tfidf_data.columns[1:], no_top_words=4)
print("\nNMF Model Topics:\n")

if __name__ == "__main__":
    # Tokenize the original text
    tokenized_texts = data['Reports'].dropna().apply(lambda x: x.split()).tolist()

    # Create dictionary and corpus
    from gensim.corpora import Dictionary
    id2word = Dictionary(tokenized_texts)
    corpus = [id2word.doc2bow(text) for text in tokenized_texts]

    # Extract top topic words
    lda_topic_words = [
        [count_data.columns[1:][i] for i in topic.argsort()[-4:][::-1]]
        for topic in lda_model.components_
    ]

    nmf_topic_words = [
        [tfidf_data.columns[1:][i] for i in topic.argsort()[-4:][::-1]]
        for topic in nmf_model.components_
    ]

    # Define coherence function
    from gensim.models import CoherenceModel

    def coherence_score(topics, texts, dictionary, coherence='c_v'):
        cm = CoherenceModel(topics=topics, texts=texts, dictionary=dictionary, coherence=coherence)
        return cm.get_coherence()

    # Calculate coherence
    lda_coherence = coherence_score(lda_topic_words, tokenized_texts, id2word)
    nmf_coherence = coherence_score(nmf_topic_words, tokenized_texts, id2word)

    print(f"\nLDA Coherence Score: {lda_coherence:.4f}")
    print(f"NMF Coherence Score: {nmf_coherence:.4f}")


#topic name assignment
lda_model_labels = {
    0: "Access & Online learning issues",
    1: "Career & university Opportunities",
    2: "Academic Stress & well being",
    3: "Campus facilities & services",
}

nmf_model_labels = {
    0: "Access & online barriers",
    1: "Student Financial Struggles",
    2: "Emotional & Academic Stress",
    3: "University & campus experience",
}

#Getting dominat topic (highest probability) for each document
data['LDA_Topic'] = lda_topics.argmax(axis=1)
data['NMF_Topic'] = nmf_topics.argmax(axis=1)

# Mapping the topic labels to the original data
data['LDA_Topic_Label'] = data['LDA_Topic'].map(lda_model_labels)
data['NMF_Topic_Label'] = data['NMF_Topic'].map(nmf_model_labels)

# Saving the data with topics
data.to_csv('data_with_topics.csv', index=False)


