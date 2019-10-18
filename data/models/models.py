'''
# forked from https://github.com/ermongroup/WikipediaPovertyMapping.git
    Evan Sheehan
    Project: Stanford AI Lab, Stefano Ermon Group & Computational Sustainability, Wiki-Satellite
    Loads the array of coordinate articles and trains a doc2vec model on them
'''

import sys
# from data_processor import *
import gensim
from gensim.models.doc2vec import TaggedDocument
import os
import collections
import random
import multiprocessing
from util_corpora import *
import datetime

smokescreen = True

# old code from previous year that does not run currently without data_processor module

# def read_corpus(array):
#     data = []
#     for i in array:
#         #if "Body Text" in array[i][6]:
#             #doc = array[i][6]["Body Text"]
#         #else:
#             #doc = array[i][1]
#         doc = get_text(i)
#         data.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [get_title(i)]))
#     return data

# # Load all coordinate articles across all categories, including Uncategorized ones
# wiki_train_file = load_coordinate_array("full", uncategorized=True,  verbose=True)
# train_corpus = read_corpus(wiki_train_file)
# print("Data formatted")


corpus_name = 'wiki-english-20171001'

if smokescreen:
    import gensim.test.utils
    corpus = gensim.test.utils.common_texts
    train_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
else:
    corpus = load_corpus(corpus_name)

    # for dump processed with 'python -m gensim.scripts.segment_wiki' having structure
    # (str, list of (str, str), (Optionally) dict of (str: str))
    #         Structure contains (title, [(section_heading, section_content), ...], (Optionally) {interlinks}).
    train_corpus = [TaggedDocument([doc[i][0]].split(' ') + [' '] + [(doc[i][1][j][0] + ' ' + doc[i][1][j][1]).split(' ')
                                    for j in range(len(doc[i][1]))], [i])
                                    for i, doc in enumerate(corpus)]
print("Data formatted")

# Instantiate the model and build the vocabulary
dbow_words = 1
dm = 0
vector_size = 300
window = 8
min_count = 15
epochs = 10
if smokescreen:
    dbow_words = 1
    dm = 0
    vector_size = 5
    window = 2
    min_count = 3
    epochs = 1

# DBOW, with word vectors as well
model = gensim.models.doc2vec.Doc2Vec(dm=dm, dbow_words=dbow_words,
                                      vector_size=vector_size, window=window, min_count=min_count, epochs=epochs,
                                      workers=multiprocessing.cpu_count())
# model = gensim.models.doc2vec.Doc2Vec(documents=articles, dm=0, dbow_words=1, vector_size=300, window=8, min_count=15, epochs=10, workers=multiprocessing.cpu_count())

# DM
#model = gensim.models.doc2vec.Doc2Vec(vector_size=1000, window=8, min_count=15, epochs=50, workers=multiprocessing.cpu_count())

print(train_corpus)
model.build_vocab(train_corpus)

print("Vocabulary built")
if __name__ == '__main__':
    # Train the model
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("Model trained")

    start_time = datetime.datetime.strftime("%b_%d_%Y_%H:%M:%S")

    # Save the model
    model.save('./wikimodel_DBOW{}_DM{}_vector{}_window{}_count{}_epoch{}__' + start_time + '.doc2vec'
               .format(dbow_words, dm, vector_size, window, min_count, epochs))
    print("Model saved")
