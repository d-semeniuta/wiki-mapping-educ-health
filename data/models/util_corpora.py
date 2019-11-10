import gensim.downloader as downloader
import os
import json
from smart_open import smart_open

# example of loading data in gensim
# see gensim.downloader.info() for models and corpora available for download
# see gensim.downloader.info()['corpora'] for corpora specifically
# 'wiki-english-20171001' for Wikipedia data set

# downloads corpus from gensim-data repo and stores corpus path to corpus_path__<corpus_name>.txt
def gensim_download_data(corpus_name):
    corpus_path = downloader.load(corpus_name, return_path=True)
    with open('corpus_path__{}.txt'.format(corpus_name), 'w') as f:
        f.write(corpus_path)
    return corpus_path

# loads corpus at path, internal function
def gensim_load_data(path):
    with smart_open(path, 'rb', encoding="utf-8") as infile:
    	return json.load(infile)

# loads downloaded corpus corpus_name or else downloads the corpora and then loads it
def load_corpus(corpus_name):
    if not os.path.exists('corpus_path__{}.txt'.format(corpus_name)):
        gensim_download_data(corpus_name)
    with open('corpus_path__{}.txt'.format(corpus_name), 'r') as f:
        path = f.read()

    return gensim_load_data(path)

if __name__ == '__main__':
    corpus_name = 'semeval-2016-2017-task3-subtaskBC'
    # corpus_name = 'wiki-english-20171001'

    corpus = load_corpus(corpus_name)
    print(corpus)

