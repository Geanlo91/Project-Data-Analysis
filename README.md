# Topic Modelling on Student Reports
This project uses Natural Language Processing (NLP) and unsupervised machine learning to perform topic modeling on a collection of student-written reports. 
The goal is to uncover hidden themes within the texts and gain insight into student concerns, experiences, and needs.

## Project Structure

├── Datasetprojpowerbi.csv         # Original dataset

├── preprocessed_data.csv          # Cleaned and lemmatized text

├── count_data.csv                 # Count Vectorized representation (Bag-of-Words)

├── tfidf_data.csv                 # TF-IDF representation

├── data_with_topics.csv           # Data annotated with topic numbers and labels

├── topic_extraction.py            # Main code for topic modeling and visualization

└── README.md                      # Project documentation

## Features
Text cleaning: lowercasing, punctuation and number removal, stopwords filtering, lemmatization.

Vectorization using both CountVectorizer and TfidfVectorizer.

Topic modeling with:
-  Latent Dirichlet Allocation (LDA) for BoW.
-  Non-negative Matrix Factorization (NMF) for TF-IDF.

Topic coherence scoring to evaluate model quality.

Topic visualization with:
- Word clouds.
- Bar charts.
- Topic distribution graphs.

Semantic topic labeling for human interpretability.

## How to run the project
1. Clone the Repository
2. Install dependencies
- pip install -r requirements.txt or manually
- pip install pandas numpy matplotlib seaborn nltk gensim scikit-learn workcloud
   
3. Run the code
You can run the full workflow from preprocessing to visualization using a Python script (e.g topic_extraction.py) or step through it in a Jupyter notebook.
