from wikiapi import WikiApi
import gensim
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def get_relevant_articles(keywords, search_depth=5, keyword_summary=3, min_com_words=1):
    """
    Searches through a list of keywords and returns keywords based on article headers
    in Wikipedia.

    args:
    *  keywords: A list of keywords
    *  search_depth: how many wikipedia search results are checked, assumes to be between 1-10
    *  keyword_summary: gensim word argument to how many words should be used in summarization
    """
    if len(keywords) == 0:
        return []
    wiki = WikiApi()

    keywords = [x.lower() for x in keywords]
    info = []
    for keyword in keywords:
        results = wiki.find(keyword)
        other_words = [x for x in keywords if x != keyword]

        if search_depth is not None:
            results = results[:search_depth]

        for result in results:
            article = wiki.get_article(result)
            summary_words = article.summary.lower().split(' ')
            #has_words = any(word in summary_words for word in other_words)
            nr_of_common_words = 0
            for word_in_others in other_words:
                 for word_in_summary in summary_words:
                     if word_in_others == word_in_summary:
                         nr_of_common_words = nr_of_common_words + 1

            if nr_of_common_words >= min_com_words:
                 info.append(article.heading)
                 print("info: " + str(info))

            # if has_words:
            #     info.append(article.heading)
            #     print("info: " + str(info))

    try:
        info_keyword = gensim.summarization.keywords(' '.join(info),
                                                     words=keyword_summary).split('\n')
    except:
        print("keyword extraction failed, defaulting to article heading output")
        info_keyword = info[:]
    return info_keyword

#all_results, has_words, info, article = get_relevant_articles(
#    "stock market investor fund trading investment firm exchange companies share".split())

def remove_duplicates(input):
    seen = set()
    result = []
    for item in input:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def lemmatize_all(docs):
    """lemmatize a list of strings"""
    #from gensim import utils
    #import itertools
    #def lemmatize_single(doc):
        #result = utils.lemmatize(doc)
        #return [x[:-3] for x in result]

    return [lemmatizer.lemmatize(x) for x in docs]


#print(all_results)  # [u'investor', u'investors', u'reform', u'investment', u'exchange', u'trade', u'trading']
#print(lemmatize_all(all_results))  # ['exchange', 'reform', 'trade', 'trading', 'investor', 'investment']


