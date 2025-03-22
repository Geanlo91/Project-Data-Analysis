from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
from sklearn.decomposition import LatentDirichletAllocation, NMF
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def compute_coherence_scores(model_type, data_matrix, feature_names, texts, dictionary, topic_range):
    scores = []
    for k in topic_range:
        print(f"Training {model_type} with {k} topics...")
        
        if model_type == 'lda':
            model = LatentDirichletAllocation(n_components=k, random_state=42)
        else:
            model = NMF(n_components=k, random_state=42)

        model.fit(data_matrix)

        # Extract only 4 top words per topic
        top_words = [
            [feature_names[i] for i in topic.argsort()[-5:][::-1]]
            for topic in model.components_
        ]
        
        # Ensure selected words exist in dictionary
        top_word_ids = [
            [dictionary.token2id[word] for word in topic if word in dictionary.token2id]
            for topic in top_words
        ]

        coherence_model = CoherenceModel(topics=top_word_ids, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        
        print(f"Coherence Score for {k} topics: {coherence_score}")
        scores.append(coherence_score)
    
    return scores

if __name__ == "__main__":
    # Load data
    data = pd.read_csv('preprocessed_data.csv')
    count_data = pd.read_csv('count_data.csv')
    tfidf_data = pd.read_csv('tfidf_data.csv')

    # Tokenize text and create dictionary
    tokenized_texts = data['Reports'].dropna().apply(lambda x: x.split()).tolist()
    id2word = Dictionary(tokenized_texts)
    corpus = [id2word.doc2bow(text) for text in tokenized_texts]

    # Define topic range (loop over multiple numbers)
    topic_range = list(range(2, 11))  # Testing from 2 to 10 topics

    # Compute coherence scores for LDA and NMF
    lda_scores = compute_coherence_scores('lda', count_data.iloc[:, 1:].values, count_data.columns[1:], tokenized_texts, id2word, topic_range)
    nmf_scores = compute_coherence_scores('nmf', tfidf_data.iloc[:, 1:].values, tfidf_data.columns[1:], tokenized_texts, id2word, topic_range)

    # Print results
    print(f"\nTopic Range: {topic_range}")
    print(f"LDA Scores: {lda_scores}") 
    print(f"NMF Scores: {nmf_scores}")

    # Plot coherence scores
    plt.figure(figsize=(10, 6))
    plt.plot(topic_range, lda_scores, label='LDA', marker='o')
    plt.plot(topic_range, nmf_scores, label='NMF', marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score (c_v)")
    plt.title("Coherence Score vs Number of Topics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

