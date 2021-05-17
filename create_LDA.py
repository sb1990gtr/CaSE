from sklearn.datasets import fetch_20newsgroups

from src.lime_text_topics import PreProcessor

dataset_complete_only_text = fetch_20newsgroups()
data_only = dataset_complete_only_text.data
targets_only = dataset_complete_only_text.target

# Preprocess dataset
# includes removing of unnecessary characters and lemmatization

data_processor = PreProcessor()
data_preprocessed = data_processor.preprocess_dataset(data_only)

import gensim
import gensim.corpora as corpora

# Create Dictionary
id2word = corpora.Dictionary(data_preprocessed)
id2word.save_as_text('20_newsgroup_dict')
# create corpus
corpus = [id2word.doc2bow(text) for text in data_preprocessed]
# create LDA
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=1000,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

lda_model.save('lda_20newsgroup_20topics.model')