import logging
import gensim

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


id2word = gensim.corpora.Dictionary.load_from_text('wikicorp/_wordids.txt.bz2')
# load corpus iterator
mm = gensim.corpora.MmCorpus('wikicorp/_tfidf.mm')
print(mm)

lda = gensim.models.ldamodel.LdaModel(corpus=mm,
                                      id2word=id2word,
                                      num_topics=100,
                                      update_every=1,
                                      passes=1)
lda.print_topics(20)

lda.save('lda_wiki_100topics.model')
