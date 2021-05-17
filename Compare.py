import sklearn.metrics
import sklearn.feature_extraction
import pandas as pd
from sklearn import preprocessing
import sys, os
import numpy

print('The scikit-learn version is {}.'.format(sklearn.__version__))

from nltk.corpus import reuters

label_topics="NoLabel" #possible values: "LabelAndSave", "Load", "LoadGoogle", "NoLabel", "LabelAndSaveGoogle"
le = preprocessing.LabelEncoder()
base_path = "/Users/sb1990gtr/Documents/Basti/Promotion/Uni/Veröffentlichung_2/LDA_LIME/R52_Experiment/src/lime_topic_final/"

sys.path.append(os.path.abspath(base_path))
import AutomatedTopicLabeling as topiclabeler

dataset_test_all_terms = pd.read_csv(base_path + "r52-test-all-terms.txt", delimiter='\t', header=None)

dataset_train = pd.read_csv(base_path + "r52-train-no-stop.txt", delimiter='\t', header=None)
dataset_train.columns = ["labels", "features"]
train_labels_numerical = le.fit(dataset_train.labels)
train_labels_numerical.classes_

dataset_test = pd.read_csv(base_path + "r52-test-no-stop.txt", delimiter='\t', header=None)
dataset_test.columns = ["labels", "features"]

dataset_complete = pd.read_csv(base_path + "r52-complete.txt", delimiter='\t', header=None)
dataset_complete.columns = ["labels", "features"]
all_labels_unique = pd.unique(dataset_complete.labels)

class_names = [label for label in train_labels_numerical.classes_]


#vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
vectorizer = sklearn.feature_extraction.text.CountVectorizer()
train_vectors = vectorizer.fit_transform(dataset_train.features)
test_vectors = vectorizer.transform(dataset_test.features)
all_vectors = vectorizer.transform(dataset_complete.features)

#from sklearn import tree
#clf = tree.DecisionTreeClassifier(criterion = 'entropy').fit(train_vectors, dataset_train.labels)

#from sklearn.naive_bayes import MultinomialNB
#nb = MultinomialNB(alpha=.01)
#clf = nb.fit(train_vectors, dataset_train.labels)


#from sklearn.linear_model import LogisticRegression
#clf = LogisticRegression(random_state=0).fit(train_vectors, dataset_train.labels)
#coef_log_reg = clf.coef_.T
#class_names = clf.classes_
#feature_names = vectorizer.get_feature_names()
#coeff_df = pd.DataFrame(coef_log_reg, columns=class_names, index=feature_names)

# SGBoost
import xgboost as xgb
xg = xgb.XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=10,
                              silent=True, objective='reg:logistic', booster='gbtree')
clf = xg.fit(train_vectors, dataset_train.labels)

#from sklearn.svm import SVC
#clf = SVC(gamma='auto', probability=True, kernel="linear")
#clf.fit(train_vectors, dataset_train.labels)

#predict & evaluate
pred = clf.predict(test_vectors)
sklearn.metrics.f1_score(dataset_test.labels, pred, average='weighted')


from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, clf)


# choose the instance
dataset_test[dataset_test["labels"] == 'acq']
idx = 371#1081#95
# used in paper: 645#566 auch spannend #456  #345 best #332 guter Vergleich, zeigt mental calc und interaction ala short term  #321
print("Original instance: " + str(dataset_test_all_terms[1][idx]))
print("Original instance label: " + str(dataset_test_all_terms[0][idx]))



#exp.as_pyplot_figure(label=exp.available_labels()[0])
#exp.save_to_file('lime_word.html', labels=[exp.available_labels()[0]])

#print ('Explanation for class %s' % class_names[0])
#print ('\n'.join(map(str, exp.as_list(label=0))))
# print ()
# print ('Explanation for class %s' % class_names[17])
# print ('\n'.join(map(str, exp.as_list(label=17))))

#exp.show_in_notebook(text=False, lbels=(0,))

import sys, os
sys.path.append("/Users/sb1990gtr/Documents/Basti/Promotion/Uni/Veröffentlichung_2/LDA_LIME/R52_Experiment/src/")

from lime_topic_final.lime_text_topics import PreProcessor

# Preprocess document within dataset
# includes removing of unnecessary characters and lemmatization

data_processor = PreProcessor()
data_preprocessed = data_processor.preprocess_string(dataset_test.features[idx])

#print('\n--- the origional document---\n')
#print(dataset_test.features[idx])
#print('\n--- the preprocessed document---\n')
#print(data_preprocessed)

import gensim
from gensim.test.utils import datapath
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Loading
temp_file = datapath(base_path + "lda_reuters_19_final.model")
# Load a potentially pretrained model from disk.
lda_model = LdaModel.load(temp_file)

#prüfen, ob topics gelabeld werden sollen
if (label_topics == "LabelAndSave"):
    topics = []
    for topic_id, topic in enumerate(lda_model.show_topics(num_topics=lda_model.num_topics)):
        most_imp_words = []
        words_prob_per_topic = lda_model.show_topic(topic[0], topn=10)
        for word, proba in words_prob_per_topic:
            #print(str(word))
            most_imp_words.append(word)

        #print(str(topic_id), str(lda_model.show_topic(topic_id, topn=10)[0]))
        topics_labels = topiclabeler.remove_duplicates(topiclabeler.lemmatize_all(topiclabeler.get_relevant_articles(most_imp_words)))
        topics.append('topic #'+str(topic[0]) + ": " + str(topics_labels))
        #save topics_list
        f = open('topics' + str(lda_model.num_topics) + '.txt', 'w')
        for ele in topics:
            f.write(ele + '\n')

        f.close()

elif (label_topics=="LabelAndSaveGoogle"):
    import AutomatedGoogleTopicLabeling as googleLabeler
    topics = []
    for topic_id, topic in enumerate(lda_model.show_topics(num_topics=lda_model.num_topics)):
        most_imp_words = []
        words_prob_per_topic = lda_model.show_topic(topic[0], topn=5)
        for word, proba in words_prob_per_topic:
            # print(str(word))
            most_imp_words.append(word)

        # print(str(topic_id), str(lda_model.show_topic(topic_id, topn=10)[0]))
        topics_labels = googleLabeler.pipeline(most_imp_words)

        topics.append('topic #' + str(topic[0]) + ": " + str(topics_labels))
        # save topics_list
        f = open('topics_google' + str(lda_model.num_topics) + '.txt', 'w')
        for ele in topics:
            f.write(ele + '\n')

        f.close()

elif (label_topics=="Load"):
    topics = open('topics' + str(lda_model.num_topics) + '.txt', "r")
    print(topics.read())

elif (label_topics=="LoadGoogle"):
    topics = open('topics_google' + str(lda_model.num_topics) + '.txt', "r")
    print(topics.read())

else:
    topics = []
    for x in range(lda_model.num_topics):
        topics.append('topic #'+str(x))

# and load the dictionary for the LDA_model
id2word = Dictionary.load_from_text(base_path + 'dict.txt')


#print(f'Number of topics: {lda_model.num_topics}')
# print top words for one topic
#lda_model.print_topics(1)

# minimum_probability = 0.001
#
#
# def word_to_topics(word):
#     """
#     Maps a word on its corresponding topics
#     :param word: the word that is searched for
#     :return: list of topics
#     """
#
#     word_lem_list = data_processor.lemmatize(word)
#     if len(word_lem_list) > 0:
#         word_lem = word_lem_list[0]
#         if word_lem in id2word.token2id:
#             word_id = id2word.token2id[word_lem]
#             return [x[0] for x in lda_model.get_term_topics(word_id, minimum_probability=minimum_probability)]
#         else:
#             return []
#     else:
#         return []

def word_to_topics(word):
    """
    Maps a word on its corresponding topics
    :param word: the word that is searched for
    :return: list of topics
    """
    #word_lem_list = data_processor.lemmatize(word)
    #if len(word_lem_list) > 0:
        #word_lem = word_lem_list[0]
        if word in id2word.token2id:
                word_id = id2word.token2id[word]
                # so im paper ecml verwernet: if lda_model.get_term_topics(word_id, minimum_probability=0.000000000000001):
                if lda_model.get_term_topics(word_id, minimum_probability=0.000000000000001):
                        #so in ECML verwendet z = [x for x in lda_model.get_term_topics(word_id, minimum_probability=0.0000000000000000001)]
                        z = [x for x in lda_model.get_term_topics(word_id, minimum_probability=0.0000000000000000001)]
                        z_sorted = sorted(z, key=lambda tupel:tupel[1], reverse=True)
                        #print(str([z[0] for z in z_sorted[:1]]))
                        return [z[0] for z in z_sorted[:1]]

                else:
                    #so in ECML verwerndet z = [x for x in lda_model.get_term_topics(word_id, minimum_probability=0.0000000000000000003)]
                    z = [x for x in lda_model.get_term_topics(word_id, minimum_probability=0.0000000000000000003)]
                    z_sorted = sorted(z, key=lambda tupel: tupel[1], reverse=True)
                    #print(str([z[0] for z in z_sorted[:1]]))
                    return [z[0] for z in z_sorted[:1]]
        else:
            return[]
   # else:
        #eturn[]


from lime.lime_text import LimeTextExplainer
from lime_topic_final.lime_text_topics import LimeTextByTopicsExplainer
sys.path.append("/Users/sb1990gtr/Documents/Basti/Promotion/Uni/Veröffentlichung_2/LDA_LIME/R52_Experiment/src/lime_topic_final/")
from lime_topic_final.lime_topic_words import LimeTextByTopicSampledWordsExplainer

# explain the instance using word-based standard LIME
explainer = LimeTextExplainer(class_names=class_names, feature_selection="forward_selection", random_state=54321, verbose=False)
exp = explainer.explain_instance(dataset_test.features[idx], c.predict_proba, num_features=80, top_labels=1)

# explain the instance using word-based LIME-SFA
explainer_topic_words = LimeTextByTopicSampledWordsExplainer(class_names=class_names, consider_all_words=True, word_to_topics=word_to_topics, topics=topics, feature_selection="forward_selection", random_state=54321, verbose=False)
exp_topic_words = explainer_topic_words.explain_instance(dataset_test.features[idx], c.predict_proba, num_features=12, top_labels=1)

# explain the instance using topicLIME
explainer_mod = LimeTextByTopicsExplainer(class_names=class_names, consider_all_words=False, word_to_topics=word_to_topics, topics=topics, feature_selection="forward_selection", random_state=54321, verbose=False)
exp_mod = explainer_mod.explain_instance(dataset_test.features[idx], c.predict_proba, num_features=8, top_labels=1)

#instrinsic explanations for Logistic Regression
def extractMostImportantWords(noOfWords=10):
    #get label for selected predicted instance
    label_pred = clf.predict(test_vectors[idx])[0]
    #match words from instance with words from corresponding ovr model
    words_from_instance = pd.DataFrame(str(dataset_test.features[idx]).split(), columns=["word"])
    words_from_ovr_model = coeff_df.loc[:,label_pred]
    words_from_instance.index.name='word_index'
    relevant_words = words_from_ovr_model.loc[coeff_df.index.isin(words_from_instance["word"]),]
    relevant_words_sorted = relevant_words.sort_values(ascending=False)[:noOfWords]
    return relevant_words_sorted

#print("intercept_mod : " + str(exp_mod.intercept))
#print("local prediction_mod: " + str(exp_mod.local_pred))
#print("prediction score (R square)_mod: " + str(exp_mod.score))

#Show different explanations
print('Document id: %d' % idx)
print('Predicted class =', clf.predict(test_vectors[idx]))
print('True class: %s\n' % dataset_test.labels[idx])


print('\n\n********************* Intrinsic ***********************\n\n')
extractMostImportantWords(12)


print('\n\n********************* Standard LIME ***********************\n\n')
for x in exp.available_labels():
    print(str(x))
    print('Word-based explanation for class %s' % class_names[x])
    print('\n'.join(map(str, exp.as_list(label=x))))
    print()


print('\n\n********************* LIME-SFA ***********************\n\n')
for x in exp_topic_words.available_labels():
    print('Topic-word explanation for class %s' % class_names[x])
    print('\n'.join(map(str, exp_topic_words.as_list(label=x))))
    print()


print('\n\n********************* topicLIME ***********************\n\n')
for x in exp_mod.available_labels():
    print('Topic-based explanation for class %s' % class_names[x])
    print('\n'.join(map(str, exp_mod.as_list(label=x))))
    print()


# print topic distribution of concerned document
print(str(lda_model.get_document_topics(id2word.doc2bow(dataset_test.features[idx].split()))))


########################## explanation and fidelity evaluation - topicLIME vs. LIME vs. LIME-SFA ###################################
def classifyWOExplUnit_words(instance, explanation_unit):
    # create vector representation for given instance
    # instance_vector = vectorizer.transform(pd.Series(dataset_test.features[idx]))
    instance_vector = vectorizer.transform(pd.Series(instance))

    # classify instance and calc confidence for predicted label
    classification_orig = clf.predict(instance_vector)
    classification_orig_conf = clf.predict_proba(instance_vector)
    pred_class_orig_conf = numpy.amax(classification_orig_conf)

    # prepare and create instance that does not contain the explanation units
    #orig_instance_split = dataset_test.features[idx].split()
    orig_instance_split = instance.split()

    # explanation_unit = ["ship", "cranes", "development","crane"]
    # explanation_unit = ["port"]
    document_wo_eu = [word for word in orig_instance_split if word.lower() not in explanation_unit]
    document_wo_eu_str = ' '.join(document_wo_eu)

    # create vector representation for newly created instance
    new_doc_vec = vectorizer.transform(pd.Series(document_wo_eu_str))

    # classify newly created instance
    classification_new_doc = clf.predict(new_doc_vec)
    classification_new_doc_conf = clf.predict_proba(new_doc_vec)
    # clf.classes_ are the classes that the classifier used to learn in correct order for displaying and representation
    pred_class_conf = numpy.amax(classification_new_doc_conf)
    # that the position of the most probable label from predict probab : numpy.argmax(classification_new_doc_conf)

    # calculate difference in prediction confidence for given class
    conf_diff = pred_class_orig_conf - pred_class_conf
    return conf_diff


def classifyWOExplUnit_topics(instance, explanation_unit):
    # create vector representation for given instance
    #instance_vector = vectorizer.transform(pd.Series(dataset_test.features[idx]))
    instance_vector = vectorizer.transform(pd.Series(instance))
    #explanation_unit = ['rubbermaid']
    # classify instance and calc confidence for predicted label
    classification_orig = clf.predict(instance_vector)
    classification_orig_conf = clf.predict_proba(instance_vector)
    pred_class_orig_conf = numpy.amax(classification_orig_conf)

    # prepare and create instance that does not contain the explanation units
    #orig_instance_split = dataset_test.features[idx].split()
    orig_instance_split = instance.split()

    # explanation_unit = ["ship", "cranes", "development","crane"]
    # explanation_unit = ["port"]
    document_wo_eu = [word for word in orig_instance_split if word.lower() not in explanation_unit]
    document_wo_eu_str = ' '.join(document_wo_eu)

    # create vector representation for newly created instance
    new_doc_vec = vectorizer.transform(pd.Series(document_wo_eu_str))

    # classify newly created instance
    classification_new_doc = clf.predict(new_doc_vec)
    classification_new_doc_conf = clf.predict_proba(new_doc_vec)
    # clf.classes_ are the classes that the classifier used to learn in correct order for displaying and representation
    pred_class_conf = numpy.amax(classification_new_doc_conf)
    # that the position of the most probable label from predict probab : numpy.argmax(classification_new_doc_conf)

    # calculate difference in prediction confidence for given class
    conf_diff = pred_class_orig_conf - pred_class_conf
    return conf_diff



# parameter definitions for eval run #
# n_eu = 4 # the number of LIME topic explanation units that shall be checked
# # TODO: Permutiere a) Anzahl n_eu, b) Anzahl der Topics aus dem LDA Model und c) den BASE Classifier d) ziehe UNKNOWN words hinzu
#
#
def eval_topics(n_eu, instance):
    # definiere und setze Gesamtfehler_topic auf 0
    # # für jede eu in n_eu aus LIME topic Erklärung
    #build explanation object
    exp_mod = explainer_mod.explain_instance(instance, c.predict_proba, num_features=n_eu, top_labels=1)
    #exp_mod = explainer_mod.explain_instance(dataset_test.features[idx], c.predict_proba, num_features=10, top_labels=1)

    for x in exp_mod.available_labels():
        explanations_topic = exp_mod.as_list(label=x)

    n_explanations_topic = explanations_topic[:n_eu]
    conf_difference_topic_local = 0
    eus = []
    for expl in n_explanations_topic:
        explanation_unit = (expl[0][expl[0].find('[')+1:expl[0].find(']')])
        eu = explanation_unit.replace("'","").replace(" ","").split(',')
        #conf_difference_topic = classifyWOExplUnit_topics(dataset_test.features[idx], eu)
        conf_difference_topic = classifyWOExplUnit_topics(instance, eu)
        conf_difference_topic_local += conf_difference_topic
        eus.extend(eu)
    num_words_in_topic = len(eus)


    return conf_difference_topic_local, num_words_in_topic
#
def eval_words(num_words, instance):
    # definiere und setze Gesamtfehler_wort auf 0
    # # für jede eu in n_eu aus LIME wort Erklärung
    # build explanation object for according instance
    exp = explainer.explain_instance(instance, c.predict_proba, num_features=num_words, top_labels=1)
    #exp = explainer.explain_instance(dataset_test.features[idx], c.predict_proba, num_features=40, top_labels=1)

    for x in exp.available_labels():
        explanations_word = exp.as_list(label=x)

    #n_explanations_word = explanations_word[:num_words_in_topic]
    n_explanations_word = explanations_word[:num_words]
    conf_difference_word_local = 0
    for expl in n_explanations_word:
        conf_difference_word = classifyWOExplUnit_words(instance, [expl[0]])
        conf_difference_word_local += conf_difference_word
    #conf_difference_word = classifyWOExplUnit_words(dataset_test.features[idx], [expl[0]])
    return conf_difference_word_local

#
# #run eval
# difference_topic_total = 0
# difference_word_total = 0
#
# for instance in dataset_test.features:
#     conf_difference_topic, num_words_in_topic = eval_topics(instance)
#     #mean_error_topic_instance, num_words_in_topic = eval_topics(dataset_test.features[idx])
#
#     difference_topic_total += conf_difference_topic
#
#     difference_word_instance = eval_words(num_words_in_topic, instance)
#     #mean_error_word_instance = eval_words(num_words_in_topic, dataset_test.features[idx])
#
#     difference_word_total += difference_word_instance
#
# mean_difference_topic_total = difference_topic_total / len(dataset_test.features)
# mean_difference_word_total = difference_word_total / len(dataset_test.features)

########################## explanation and fidelity evaluation - topic_words ###################################
def classifyWOExplUnit_topic_words(instance, explanation_unit):
    # create vector representation for given instance
    # instance_vector = vectorizer.transform(pd.Series(dataset_test.features[idx]))
    instance_vector = vectorizer.transform(pd.Series(instance))

    # classify instance and calc confidence for predicted label
    classification_orig = clf.predict(instance_vector)
    classification_orig_conf = clf.predict_proba(instance_vector)
    pred_class_orig_conf = numpy.amax(classification_orig_conf)

    # prepare and create instance that does not contain the explanation units
    #orig_instance_split = dataset_test.features[idx].split()
    orig_instance_split = instance.split()

    # explanation_unit = ["ship", "cranes", "development","crane"]
    # explanation_unit = ["port"]
    document_wo_eu = [word for word in orig_instance_split if word.lower() not in explanation_unit]
    document_wo_eu_str = ' '.join(document_wo_eu)

    # create vector representation for newly created instance
    new_doc_vec = vectorizer.transform(pd.Series(document_wo_eu_str))

    # classify newly created instance
    classification_new_doc = clf.predict(new_doc_vec)
    classification_new_doc_conf = clf.predict_proba(new_doc_vec)
    # clf.classes_ are the classes that the classifier used to learn in correct order for displaying and representation
    pred_class_conf = numpy.amax(classification_new_doc_conf)
    # that the position of the most probable label from predict probab : numpy.argmax(classification_new_doc_conf)

    # calculate difference in prediction confidence for given class
    conf_diff = pred_class_orig_conf - pred_class_conf
    return conf_diff


#äparameter definitions for eval run #
#number_of_words = 10 # the number of LIME topic_words explanation units that shall be checked
n_eu = 5

def eval_topic_words(number_of_words, instance):
    # definiere und setze Gesamtfehler_wort auf 0
    # build explanation object for according instance
    exp_top_words = explainer_topic_words.explain_instance(instance, c.predict_proba, num_features=number_of_words, top_labels=1)
    #exp = explainer.explain_instance(dataset_test.features[idx], c.predict_proba, num_features=40, top_labels=1)

    for x in exp_top_words.available_labels():
        explanations_topic_words = exp_top_words.as_list(label=x)

    #n_explanations_word = explanations_word[:num_words_in_topic]
    n_explanations_topic_word = explanations_topic_words[:number_of_words]
    conf_difference_topic_words_local = 0
    for expl in n_explanations_topic_word:
        conf_difference_topic_words = classifyWOExplUnit_topic_words(instance, [expl[0]])
        conf_difference_topic_words_local += conf_difference_topic_words
    #conf_difference_word = classifyWOExplUnit_words(dataset_test.features[idx], [expl[0]])
    return conf_difference_topic_words_local


#run eval
difference_topic_words_total = 0
difference_words_total = 0
difference_topics_total = 0


for instance in dataset_test.features:

    difference_topics, num_of_words_in_top = eval_topics(n_eu, instance)
    #difference_topic_words = eval_topic_words(num_of_words_in_top, instance)
    difference_words = eval_words(num_of_words_in_top, instance)

    difference_topics_total += difference_topics
    #difference_topic_words_total += difference_topic_words
    difference_words_total += difference_words


mean_difference_topics_total = difference_topics_total / len(dataset_test.features)
#mean_difference_topic_words_total = difference_topic_words_total / len(dataset_test.features)
mean_difference_words_total = difference_words_total / len(dataset_test.features)


# #Analyse der Local Fidelity der Explanation models generated by LIME, LIME-SFA and topicLIME
# error_word = 0
# #error_topic = 0
# error_topic_word = 0
#
# r_square_word = 0
# #r_square_topic = 0
# r_square_topic_word = 0
#
# #number_of_topics_eu = 5
# number_of_words = 20
# #mean_error_topic_total = 0
# mean_error_topic_word_total = 0
# mean_error_word_total = 0
# #mean_r2_topic_total = 0
# mean_r2_topic_word_total = 0
# mean_r2_word_total = 0
#
# for instance in dataset_test.features:
#     #get model prediction
#     instance_vector = vectorizer.transform(pd.Series(instance))
#     model_prediction = numpy.amax(clf.predict_proba(instance_vector))
#
#     # # topics first
#     # exp_mod = explainer_mod.explain_instance(instance, c.predict_proba, num_features=number_of_topics_eu, top_labels=1)
#     # local_pred_topic = exp_mod.local_pred
#     # r2_topic = exp_mod.score
#     # topic_diff = abs(local_pred_topic - model_prediction)
#     # error_topic += topic_diff
#     # r_square_topic += r2_topic
#
#     # topic-words first
#     exp_top_words = explainer_topic_words.explain_instance(instance, c.predict_proba, num_features=number_of_words, top_labels=1)
#     #local_pred_topic = exp_mod.local_pred
#     local_pred_topic_word = exp_top_words.local_pred
#
#     #r2_topic = exp_mod.score
#     r2_topic_word = exp_top_words.score
#
#     #topic_diff = abs(local_pred_topic - model_prediction
#     topic_word_diff = abs(local_pred_topic_word - model_prediction)
#
#     #error_topic += topic_diff
#     error_topic_word += topic_word_diff
#
#     # r_square_topic += r2_topic
#     r_square_topic_word += r2_topic_word
#
#     #calculate number of words in topic explanation
#     # num_words_in_topic = 0
#     # for x in exp_mod.available_labels():
#     #     explanations_topic = exp_mod.as_list(label=x)
#     #
#     # for expl in explanations_topic:
#     #     explanation_unit = (expl[0][expl[0].find('[')+1:expl[0].find(']')])
#     #     eu = explanation_unit.replace("'","").replace(" ","").split(',')
#     #     num_words_in_topic += len(eu)
#
#     #words next
#     # exp = explainer.explain_instance(instance, c.predict_proba, num_features=number_of_words, top_labels=1)
#     # local_pred_word = exp.local_pred
#     # r2_word = exp.score
#     # word_diff = abs(local_pred_word - model_prediction)
#     # error_word += word_diff
#     # r_square_word += r2_word
#
#
# #mean_error_topic_total = error_topic / len(dataset_test.features)
# mean_error_topic_word_total = error_topic_word / len(dataset_test.features)
# #mean_error_word_total = error_word / len(dataset_test.features)
#
# #mean_r2_topic_total = r_square_topic / len(dataset_test.features)
# mean_r2_topic_word_total = r_square_topic_word / len(dataset_test.features)
# # #mean_r2_word_total = r_square_word / len(dataset_test.features)