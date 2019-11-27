from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import utils
import argparse
import logging
import multiprocessing
import sys
import json
import time

logger = logging.getLogger(__name__)

def build_doc2vec(input_file_path, output_file_path, num_workers):
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    train_list = []
    count = 0
    logger.info("starting!")
    doc_iterator = (TaggedDocument(utils.simple_preprocess(json.loads(line)["text"]), json.loads(line)["title"]) for line in utils.open(input_file_path, 'rb'))
    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, window=8, min_count=15, workers=num_workers)
    model.build_vocab(doc_iterator)
    for epoch in range(10):
        logger.info("Epoch "+ str(epoch + 1))
        model.train(doc_iterator)
    #
    model.save(output_file_path)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    default_workers = max(1, multiprocessing.cpu_count() - 1)
    # build_doc2vec("./raw/wikipedia/coord_articles_extraced.bz2", "./raw/wikipedia/eric_doc2vec.model", default_workers)
    # build_vector_dataset("./raw/wikipedia/coord_articles_extraced.bz2", "./raw/wikipedia/eric_doc2vec.model", "./raw/wikipedia/vector_dataest.bz2")
    parser.add_argument('-f', '--file', help='Path to the processed wiki json dump.', required=True)
    parser.add_argument('-o', '--output', help='Path of the doc2vec model that will be saved.', required=True)
    parser.add_argument('-w', '--workers', help='Number of parallel workers for multicore systems.', default = max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))

    build_doc2vec(
        args.file,
        args.output,
        args.workers
    )

    logger.info("finished running %s", sys.argv[0])
