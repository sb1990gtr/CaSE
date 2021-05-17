
"""
Automate Testing for
"""
from datetime import datetime
# region imports
import logging
import random
import sklearn.metrics
from sklearn.datasets import fetch_20newsgroups
# use original Lime explainer
from lime.lime_text import LimeTextExplainer
from src.lime_text_topics import LimeTextByTopicsExplainer, LimeTextExplanasionsComparer
# nice visualisation implemented in help class
from src.proof_of_concept import poc_visualisation

# endregion


# region parametrization
startTime = datetime.now()

# here we define critical parameters
Param__percentage_for_word_in_topic = [0.1, 0.5, 1, 2, 5, 10, 20]
Param__number_of_topics_param = [10, 20, 50, 100, 200]
Param__topic_colors = ["blue", "green", "red", "cyan", "magenta", "yellow"]
Param_sample_size = 20
Param_comparisons_per_explanation = 100
Param_top_labels = 1

# region classifier
# region create vector model

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
# making class names shorter
class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:]) for x in
               newsgroups_train.target_names]
class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB(alpha=.01)
nb.fit(train_vectors, newsgroups_train.target)

pred = nb.predict(test_vectors)
sklearn.metrics.f1_score(newsgroups_test.target, pred, average='weighted')

from sklearn.pipeline import make_pipeline

c = make_pipeline(vectorizer, nb)

# endregion
# endregion
# endregion


# region compare accoring to parameters
num_total_comparison = len(Param__percentage_for_word_in_topic) * len(Param__number_of_topics_param)
num_done_comparisons = 0
all_data = list()
poc_visualisation.nice_progressbar(0)

# get number of distinct words
allwords= []
for data_set in newsgroups_test.data:
    allwords += data_set.split()
# count distinct words
all_words_cnt = len(list(set(allwords)))

for p_word_topic in Param__percentage_for_word_in_topic:

    row_data = list()

    for num_topics in Param__number_of_topics_param:
        # mocke the word to topic map function
        def word_to_random_topics(word: str) -> list:
            out = list()
            for x in range(num_topics):
                if p_word_topic >= random.uniform(0.0, 100.0):
                    out.append(x)
            return out


        text_topic_lime_explainer = LimeTextByTopicsExplainer(class_names=class_names,
                                                         topics=["Topic" + str(i) for i in range(num_topics)],
                                                         word_to_topics=word_to_random_topics,
                                                         consider_all_words=True)
        text_word_lime_explainer = LimeTextExplainer(class_names=class_names)

        # here we test for each dataset if proof of concept holds
        comparer = LimeTextExplanasionsComparer(text_topic_lime_explainer,
                                                num_topics,
                                                p_word_topic,
                                                text_word_lime_explainer,
                                                newsgroups_test.data,
                                                c.predict_proba)
        avg_percentage_reject_hNull, avg_word_count, av_word_count_topic_unknown = comparer.compare(Param_sample_size,
                                                                                                    Param_comparisons_per_explanation,
                                                                                                    Param_top_labels)

        row_data.append([num_topics, avg_percentage_reject_hNull, avg_word_count, av_word_count_topic_unknown])
        logging.critical("Result for " + str(p_word_topic) + " and " + str(num_topics) + " topics:\n \treject H0: " + str(
            avg_percentage_reject_hNull) + "%")

        num_done_comparisons = num_done_comparisons + 1
        poc_visualisation.nice_progressbar(num_done_comparisons / num_total_comparison)

    all_data.append([p_word_topic, row_data])
logging.critical("Final Result returned:\n" + str(
    poc_visualisation.nice_output(all_data, Param__number_of_topics_param)))
logging.critical("time taken" + str(datetime.now() - startTime))
print("\n" + str(poc_visualisation.nice_output(all_data, Param__number_of_topics_param)))

poc_visualisation.make_nice_plot(all_data, Param__number_of_topics_param, Param__topic_colors, all_words_cnt)


