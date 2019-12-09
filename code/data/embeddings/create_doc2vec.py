from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import utils
import argparse
import logging
import multiprocessing
import sys
import json
import time
​
​
def build_doc2vec(input_file_path, output_file_path, num_workers):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    train_list = []
    count = 0
    logger.info("starting!")
    doc_iterator = (TaggedDocument(utils.simple_preprocess(json.loads(line)["text"]), json.loads(line)["title"]) for line in utils.open(input_file_path, 'rb'))
    logger.info("Finished processing all of the articles.")
    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, window=8, min_count=15, workers=num_workers, epochs=10)
    model.build_vocab(doc_iterator)
    doc_iterator = (TaggedDocument(utils.simple_preprocess(json.loads(line)["text"]), json.loads(line)["title"]) for line in utils.open(input_file_path, 'rb'))
    model.train(doc_iterator, total_examples=model.corpus_count, epochs=model.epochs)
    #
    model.save(output_file_path)
​
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    default_workers = max(1, multiprocessing.cpu_count() - 1)
    parser.add_argument('-f', '--file', help='Path to the processed wiki json dump.', required=True)
    parser.add_argument('-o', '--output', help='Path of the doc2vec model that will be saved.', required=True)
    parser.add_argument('-w', '--workers', help='Number of parallel workers for multicore systems.', default = max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()
​
    logger.info("running %s", " ".join(sys.argv))
​
    build_doc2vec(
        args.file,
        args.output,
        args.workers
    )
​
    logger.info("finished running %s", sys.argv[0])
