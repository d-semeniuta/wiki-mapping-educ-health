from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim import utils
import argparse
import logging
import multiprocessing
import sys
import json

logger = logging.getLogger(__name__)

def build_doc2vec(input_file_path, output_file_path, num_workers):
    train_list = []

    count = 0

    with utils.open(input_file_path, 'rb') as file:
        for line in file:
            count += 1
            article = json.loads(line)
            text = article['text']
            title = article['title']
            train_list.append(TaggedDocument(utils.simple_preprocess(doc), [title]))
            if count % 10000 == 0:
                logger.info("Finished processing %d articles.", count)

    logger.info("Finished processing all of the articles.")
    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, window=8, min_count=15, epochs=10, workers=multiprocessing.cpu_count())

    model.build_vocab(train_list)
    logger.info("Vocab built.")

    model.train(train_list, total_examples=model.corpus_count, epochs=model.epochs)
    logger.info("Model trained.")

    model.save("./raw/wikipedia/eric_wiki.model")


if __name__ == "__main__":
    build_doc2vec(
        "./raw/wikipedia/wiki_extracted_run1_5.2mil.bz2",
        "./raw/wikipedia/eric_wiki.model",
        max(1,multiprocessing.cpu_count() - 1)
    )
