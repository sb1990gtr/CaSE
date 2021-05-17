from __future__ import unicode_literals

from functools import partial
import itertools
from collections import defaultdict
import json
import re
from pprint import pformat
from random import random

import numpy as np
import scipy as sp
import sklearn
from sklearn.utils import check_random_state

from lime import explanation
from lime import lime_base

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import spacy

import random


class TextDomainMapper(explanation.DomainMapper):
    """Maps feature ids to words"""

    def __init__(self, indexed_string):
        """Initializer.

        Args:
            indexed_string: lime_text.IndexedString, original string
        """
        self.indexed_string = indexed_string

    def map_exp_ids(self, exp, positions=False):
        """Maps ids to words or word-position strings.

        Args:
            exp: list of tuples [(id, weight), (id,weight)]
            positions: if True, also return word positions

        Returns:
            list of tuples (word, weight), or (word_positions, weight) if
            examples: ('bad', 1) or ('bad_3-6-12', 1)
        """
        exp = [(str(self.indexed_string.topic(x[0])) + ' = ' + str(self.indexed_string.words_to_topic(x[0])), x[1])
               for x in exp]
        return exp

    def visualize_instance_html(self, exp, label, div_name, exp_object_name,
                                text=True, opacity=True):
        """Adds text with highlighted words to visualization.

        Args:
             exp: list of tuples [(id, weight), (id,weight)]
             label: label id (integer)
             div_name: name of div object to be used for rendering(in js)
             exp_object_name: name of js explanation object
             text: if False, return empty
             opacity: if True, fade colors according to weight
        """
        if not text:
            return u''
        text = (self.indexed_string.raw_string()
                .encode('utf-8', 'xmlcharrefreplace').decode('utf-8'))
        text = re.sub(r'[<>&]', '|', text)
        exp_new = []
        for topic in exp:
            for word in self.indexed_string.words_to_topic(topic[0]):
                exp_new.append((word, topic[1]))
        d = defaultdict(float)
        for word, probability in exp_new:
            d[word] += float(probability)
        exp_new = [(word, probability) for word, probability in d.items()]
        exp_new = [(x[0],
                    self.indexed_string.string_position(self.indexed_string.word_to_id(x[0])),
                    x[1]) for x in exp_new]
        all_occurrences = list(itertools.chain.from_iterable(
            [itertools.product([x[0]], x[1], [x[2]]) for x in exp_new]))
        all_occurrences = [(x[0], int(x[1]), x[2]) for x in all_occurrences]
        ret = '''
            %s.show_raw_text(%s, %d, %s, %s, %s);
            ''' % (exp_object_name, json.dumps(all_occurrences), label,
                   json.dumps(text), div_name, json.dumps(opacity))
        return ret


class IndexedString(object):
    """String with various indexes."""
    bow = True

    def __init__(self, raw_string,
                 word_to_topics=None,
                 topics=None,
                 consider_all_words=False):
        """Initializer.

        Args:
            raw_string: string with raw text in it
            **************************************************************
            ** word_to_topics: function that maps a word to its topics  **
            ** topics: an array containing the names for each topic     **
            ** consider_all_words: if true non-topic words are assigned **
            **    to a fake topic                                       **
            **************************************************************
        """

        self.raw = raw_string
        self.word_to_topics = word_to_topics
        self.topics = topics

        self.split_expression = r'\W+'  # any non word character -> split into words
        splitter = re.compile(r'(%s)|$' % self.split_expression)
        self.as_list = [s for s in splitter.split(self.raw) if s]
        non_word = splitter.match

        self.as_np = np.array(self.as_list)
        self.string_start = np.hstack(
            ([0], np.cumsum([len(x) for x in self.as_np[:-1]])))
        vocab = {}
        self.inverse_vocab = []
        self.positions = []
        self.word_topics = []
        non_vocab = set()
        for i, word in enumerate(self.as_np):
            if word in non_vocab:
                continue
            if non_word(word):
                non_vocab.add(word)
                continue
            if word not in vocab:
                vocab[word] = len(vocab)
                self.inverse_vocab.append(word)
                self.positions.append([])
                self.word_topics.append(self.word_to_topics(word))
                if consider_all_words:
                    if len(self.word_topics[-1]) == 0:
                        self.word_topics[-1].append(len(self.topics) - 1)
            idx_word = vocab[word]
            self.positions[idx_word].append(i)

    def raw_string(self):
        """Returns the original raw string"""
        return self.raw

    def num_words(self):
        """Returns the number of tokens in the vocabulary for this document."""
        return len(self.inverse_vocab)

    def word(self, id_):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab[id_]

    def word_to_id(self, word):
        """Returns the word that corresponds to id_ (int)"""
        return self.inverse_vocab.index(word)

    def topic(self, id_):
        """Returns the topic name that corresponds to id_ (int)"""
        return self.topics[id_]

    def get_all_words_to_topic(self, topic_id):
        w = []
        for word_topic_cnt in range(len(self.word_topics)):
            if topic_id in self.word_topics[word_topic_cnt]:
                w.append(word_topic_cnt)
        return w

    def words_to_topic(self, _id):
        """Returns all currently present words to a topic"""
        words = []
        for w_topic, word in zip(self.word_topics, self.inverse_vocab):
            if _id in w_topic:
                words.append(word)
        return words

    def string_position(self, id_):
        """Returns a np array with indices to id_ (int) occurrences"""
        if self.bow:
            return self.string_start[self.positions[id_]]
        else:
            return self.string_start[[self.positions[id_]]]

    def inverse_removing(self, topics_to_remove):
        """Returns a string after removing the appropriate words from original string based on a given list of topics.

        Args:
            topics_to_remove: list of ids (ints) to remove

        Returns:
            original raw string with appropriate words removed.
        """
        mask = np.ones(self.as_np.shape[0], dtype='bool')
        word_mask = self.__topics_to_wordidxs(topics_to_remove)
        mask[word_mask] = False
        return ''.join([self.as_list[v] for v in mask.nonzero()[0]])

    @staticmethod
    def _segment_with_tokens(text, tokens):
        """Segment a string around the tokens created by a passed-in tokenizer"""
        list_form = []
        text_ptr = 0
        for token in tokens:
            inter_token_string = []
            while not text[text_ptr:].startswith(token):
                inter_token_string.append(text[text_ptr])
                text_ptr += 1
                if text_ptr >= len(text):
                    raise ValueError("Tokenization produced tokens that do not belong in string!")
            text_ptr += len(token)
            if inter_token_string:
                list_form.append(''.join(inter_token_string))
            list_form.append(token)
        if text_ptr < len(text):
            list_form.append(text[text_ptr:])
        return list_form

    def __topics_to_wordidxs(self, topics):
        """
        Returns the word indexes associated with the given list of topics

        :param topics: a list of topics that should be removed
        :return: list or word indexes that appear within that topic
        """
        word_idx = []
        for (word, w_topics, positions, index) in zip(self.inverse_vocab, self.word_topics, self.positions,
                                                      range(len(self.inverse_vocab))):
            if any(topic in w_topics for topic in topics):
                word_idx += positions
        return word_idx


class LimeTextByTopicsExplainer(object):
    """Explains text classifiers with topics insteda of words.
       The Explainer is build by modifying the LimeTextExplainer."""

    def __init__(self,
                 word_to_topics,
                 topics,
                 consider_all_words=False,
                 kernel_width=25,  # needed by lime default
                 kernel=None,  # needed by lime default
                 verbose=False,  # needed by lime default
                 class_names=None,  # needed by lime default
                 feature_selection='auto',  # needed by lime default
                 random_state=None):
        """Init function.

        Args:
            ***************************************************************
            **  word_to_topics: function that maps a word to its topics **
            **  topics: List of Topic  names                             **
            **  consider_all_words: if true all words that are not within**
            **        in a topic are grouped into a topic called Unknown **
            ***************************************************************
            kernel_width: kernel width for the exponential kernel.
            kernel: similarity kernel that takes euclidean distances and kernel
                width as input and outputs weights in (0,1). If None, defaults to
                an exponential kernel.
            verbose: if true, print local prediction values from linear model
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            feature_selection: feature selection method. can be
                'forward_selection', 'lasso_path', 'none' or 'auto'.
                See function 'explain_instance_with_data' in lime_base.py for
                details on what each of the options does.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """

        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        kernel_fn = partial(kernel, kernel_width=kernel_width)

        self.random_state = check_random_state(random_state)
        self.base = lime_base.LimeBase(kernel_fn, verbose,
                                       random_state=self.random_state)
        self.class_names = class_names
        self.vocabulary = None
        self.feature_selection = feature_selection

        self.word_to_topics = word_to_topics
        self.topics = topics
        self.consider_all_words = consider_all_words

        self.indexed_string = IndexedString("")

        if self.consider_all_words:
            topics.append('Unknown')

    def explain_instance(self,
                         text_instance,
                         classifier_fn,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):
        """Generates explanations for a prediction.

        First, we generate neighborhood data by randomly hiding features from
        the instance (see __data_labels_distance_mapping). We then learn
        locally weighted linear models on this neighborhood data to explain
        each of the classes in an interpretable way (see lime_base.py).

        Args:
            text_instance: raw text string to be explained.
            classifier_fn: classifier prediction probability function, which
                takes a list of d strings and outputs a (d, k) numpy array with
                prediction probabilities, where k is the number of classes.
                For ScikitClassifiers , this is classifier.predict_proba.
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity
            model_regressor: sklearn regressor to use in explanation. Defaults
            to Ridge regression in LimeBase. Must have model_regressor.coef_
            and 'sample_weight' as a parameter to model_regressor.fit()
        Returns:
            An Explanation object (see explanation.py) with the corresponding
            explanations.
        """

        self.indexed_string = IndexedString(text_instance,
                                            word_to_topics=self.word_to_topics,
                                            topics=self.topics,
                                            consider_all_words=self.consider_all_words)
        domain_mapper = TextDomainMapper(self.indexed_string)
        data, yss, distances = self.__data_labels_distances(
            self.indexed_string, classifier_fn, num_samples,
            distance_metric=distance_metric)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(self,
                                indexed_string,
                                classifier_fn,
                                num_samples,
                                distance_metric='cosine'):
        """Generates a neighborhood around a prediction.

        Generates neighborhood data by randomly removing words from
        the instance, and predicting with the classifier. Uses cosine distance
        to compute distances between original and perturbed instances.
        Args:
            indexed_string: document (IndexedString) to be explained,
            classifier_fn: classifier prediction probability function, which
                takes a string and outputs prediction probabilities. For
                ScikitClassifier, this is classifier.predict_proba.
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for sample weighting,
                defaults to cosine similarity.


        Returns:
            A tuple (data, labels, distances), where:
                data: dense num_samples * K binary matrix, where K is the
                    number of tokens in indexed_string. The first row is the
                    original instance, and thus a row of ones.
                labels: num_samples * L matrix, where L is the number of target
                    labels
                distances: cosine distance between the original instance and
                    each perturbed instance (computed in the binary 'data'
                    matrix), times 100.
        """

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        num_topics = len(self.topics)
        sample = self.random_state.randint(1, num_topics + 1, num_samples - 1)
        data = np.ones((num_samples, num_topics))
        data[0] = np.ones(num_topics)
        features_range = range(num_topics)
        inverse_data = [indexed_string.raw_string()]
        for i, size in enumerate(sample, start=1):
            inactive = self.random_state.choice(features_range, size,
                                                replace=False)
            data[i, inactive] = 0
            inverse_data.append(indexed_string.inverse_removing(inactive))
        labels = classifier_fn(inverse_data)
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances


class LimeTextExplanasionsComparer(object):
    """
    This class allows the comparison of classical Lime and the new Lime approach
    The mathematical measure test for a linear relationship between the old and new approach
        we test how often a more important topic consists of more important words, therfore we defin our hNull as
            (topic_one >= topic_two) & (topic_one_sum < topic_two_sum)
    """

    def __init__(self, topic_explainer, number_of_topics, word_in_topic_percentage, word_explainer, dataset,
                 classifier_prediction):
        """

        :param topic_explainer: a LimeTextByTopicsExplainer instance
        :param number_of_topics: the number of topics for LimeTextByTopicsExplainer
        :param word_in_topic_percentage:
        :param word_explainer: LimeTextExplainer
        :param dataset:
        :param classifier_prediction: a classifier prediction function
        """
        self.log = False
        self.lime_topic_explainer = topic_explainer
        self.number_of_topics = number_of_topics
        self.word_in_topic_percentage = word_in_topic_percentage
        self.lime_word_explainer = word_explainer
        self.dataset = dataset
        self.classifier_prediction = classifier_prediction
        if self.lime_topic_explainer.consider_all_words:
            # add additional topic which categorizes unknown words if needed
            self.number_of_topics = self.number_of_topics + 1

    def _compare_specific_explanation_all(self, lime_explanation_topic, lime_explanation_word):
        """
        comnpare all topics against each other which causes huge amount of comparisons-> num_topics!
        :param lime_explanation_topic:
        :param lime_explanation_word:
        :return: percentage of num_rejections of hNull,
        """
        topic_list = lime_explanation_topic.as_map()
        word_list = lime_explanation_word.as_map()

        num_comparisons = 0  # total amount of comparisons
        num_proofs = 0  # that H0 holds
        num_rejections = 0  # that H0 rejects

        # iterate over each label-prediction by zipping topic and word based explanations
        for (topic_pred, word_pred) in zip(topic_list.items(), word_list.items()):
            if (topic_pred[0]) != [word_pred[0]]:
                # this should never happen as we use the same classifier -> should always predict the same
                break
            lime_topic_values = topic_pred[1]
            lime_word_values = word_pred[1]
            for iter_one in range(len(lime_topic_values)):
                topic_one = lime_topic_values[iter_one]
                topic_one_sum = sum([word[1] for word in lime_word_values
                                     if word[0] in
                                     self.lime_topic_explainer.indexed_string.get_all_words_to_topic(topic_one[0])])
                for iter_two in range(iter_one + 1, len(lime_topic_values)):
                    topic_two = lime_topic_values[iter_two]
                    topic_two_sum = sum([word[1] for word in lime_word_values
                                         if word[0] in
                                         self.lime_topic_explainer.indexed_string.get_all_words_to_topic(topic_two[0])])
                    num_comparisons = num_comparisons + 1

                    if ((topic_one[1] >= topic_two[1]) & (topic_one_sum < topic_two_sum)) | (
                            (topic_two[1] >= topic_one[1]) & (topic_two_sum < topic_one_sum)):
                        num_proofs = num_proofs + 1
                    else:
                        num_rejections = num_rejections + 1

        return round(num_rejections / num_comparisons, 2) * 100

    def _compare_specific_explanation(self, lime_explanation_topic, lime_explanation_word, num_comparisons):
        """
        compares a certain amount of topics -> topics will be chosen randomly
        :param lime_explanation_topic: a lime explanation that uses topics
        :param lime_explanation_word: the classical lime explanation that uses words
        :param num_comparisons: how many Topics should be compared
        :return: percentage of num_rejections of hNull,
        """
        topic_list = lime_explanation_topic.as_map()
        word_list = lime_explanation_word.as_map()
        # test proof of concept
        # H0 and H1 -> see thesis
        _num_comparisons = 0  # total amount of comparisons
        num_proofs = 0  # that H0 holds
        num_rejections = 0  # that H0 rejects

        def _random_topic():
            return random.randint(0, self.number_of_topics - 1)

        # iterate over each label-prediction by zipping topic and word based explanations
        for (topic_pred, word_pred) in zip(topic_list.items(), word_list.items()):
            if (topic_pred[0]) != [word_pred[0]]:
                # this should never happen as we use the same classifier -> should always predict the same
                break
            lime_topic_values = topic_pred[1]
            lime_word_values = word_pred[1]
            for x in range(num_comparisons):
                one = _random_topic()
                two = _random_topic()
                topic_one = lime_topic_values[one]
                topic_one_sum = sum([abs(word[1]) for word in lime_word_values
                                     if word[0] in
                                     self.lime_topic_explainer.indexed_string.get_all_words_to_topic(topic_one[0])])
                topic_two = lime_topic_values[two]
                topic_two_sum = sum([abs(word[1]) for word in lime_word_values
                                     if word[0] in
                                     self.lime_topic_explainer.indexed_string.get_all_words_to_topic(topic_two[0])])
                # Compare both topics
                _num_comparisons = _num_comparisons + 1
                if ((abs(topic_one[1]) >= abs(topic_two[1])) & (topic_one_sum < topic_two_sum)) | (
                        (abs(topic_two[1]) >= abs(topic_one[1])) & (topic_two_sum < topic_one_sum)):
                    num_proofs = num_proofs + 1
                else:
                    num_rejections = num_rejections + 1
        return round(num_rejections / _num_comparisons, 2) * 100

    def compare(self, sample_size, comparisons_per_explanation=None, top_labels=1):
        """
        Compares different classifiers if topic explanasions
        :param sample_size: the number of samples from dataset that will be used to test it
        :param comparisons_per_explanation: number of H0 test for each label and explanation
                ->if None, all topics will be compared to each other
        :param top_labels: parameter given to Lime
        :return: avg_percentage_reject_hNull, avg_word_count, avg_word_count_topic_unknown
        """
        sum_percentage_reject_hNull = 0
        sum_word_count = 0  # which are considered by lime
        sum_word_count_topic_unknown = 0

        for x in range(sample_size):
            idx = random.randint(0, len(self.dataset) - 1)
            sum_word_count += len(self.lime_topic_explainer.indexed_string.inverse_vocab)
            if self.lime_topic_explainer.consider_all_words:
                sum_word_count_topic_unknown += len(
                    self.lime_topic_explainer.indexed_string.get_all_words_to_topic(self.number_of_topics - 1))
            exp_topic = self.lime_topic_explainer.explain_instance(self.dataset[idx],
                                                                   self.classifier_prediction,
                                                                   num_features=self.number_of_topics,
                                                                   top_labels=top_labels)

            exp_word = self.lime_word_explainer.explain_instance(self.dataset[idx],
                                                                 self.classifier_prediction,
                                                                 num_features=150,
                                                                 top_labels=top_labels)
            if comparisons_per_explanation:
                sum_percentage_reject_hNull += self._compare_specific_explanation(exp_topic, exp_word,
                                                                                  comparisons_per_explanation)
            else:
                sum_percentage_reject_hNull += self._compare_specific_explanation_all(exp_topic, exp_word)

        return round(sum_percentage_reject_hNull / sample_size), \
               round(sum_word_count / sample_size), \
               round(sum_word_count_topic_unknown / sample_size)


class PreProcessor(object):
    """
    This class holds methods for simple dataset preprocessing, ONLY for english datasets:
        - remove unnecessary characters like [@ , . ! \n ... ] etc.
        - lemmatize words (optionally use additional stemming)
        - remove stopwords from dataset
    """

    words = set(nltk.corpus.words.words())
    stop_words = stopwords.words('english')

    def __init__(self, additional_stopwords=None, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """
        simple dataset or word preprocessing using spacy, and nltk
        :param additional_stopwords: additional words that should be removed from the dataset
        :param allowed_postags: allowed psotags for natural lange processor for spacy.load('en', disable=['parser', 'ner'])
        """
        self.nlp = spacy.load('en', disable=['parser', 'ner'])
        self.allowed_postags = allowed_postags
        if additional_stopwords:
            self.stop_words.extend(additional_stopwords)

    def __lemmatize(self, word,):
        doc = self.nlp(u'' + word)
        word_lem = [token.lemma_ for token in doc if token.pos_ in self.allowed_postags]
        if len(word_lem) > 0:
            return word_lem[0]

    def lemmatize(self, sentence):
        """
        lemmatizes a sentence to a list of words using nltk
        :param sentence: a sentence
        :param stemming:
        :return: list of lemmatized words
        """
        word_tokens = word_tokenize(sentence)
        lem_words = [self.__lemmatize(word) for word in word_tokens]
        return list(filter(None, lem_words))

    def _remove_stopwords(self, words):
        """removes stopwords from sentence"""
        return [word for word in words if word not in self.stop_words]

    def _remove_unnecessary_characters(self, sentence):
        """This removes emails, new line characters and single quotes"""
        # Remove Emails
        data = re.sub('\S*@\S*\s?', '', sentence)
        # Remove new line characters
        data = re.sub('\s+', ' ', sentence)
        # Remove distracting single quotes
        data = re.sub("\'", "", sentence)
        # Remove punctuation marks
        data = re.sub(r'[^\w\s]', '', sentence)
        return data

    def _remove_nonsense_words(self, sentence):
        return " ".join(w for w in nltk.wordpunct_tokenize(sentence) if w.lower() in self.words or not w.isalpha())

    def preprocess_string(self, sentence):
        processed_string = sentence.lower()
        processed_string = self._remove_unnecessary_characters(processed_string)
        processed_string = self._remove_nonsense_words(processed_string)
        processed_string = self.lemmatize(processed_string)
        processed_string = self._remove_stopwords(processed_string)
        return processed_string

    def preprocess_dataset(self, dataset):
        """
        Returns the actual document, but each word in a lemmatized form
        :param dataset: a dataset that should be lemmatized
        :return:
        """
        return [self.preprocess_string(text) for text in dataset]
