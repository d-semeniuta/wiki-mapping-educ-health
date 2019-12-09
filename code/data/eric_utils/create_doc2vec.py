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
    train_list = []
    count = 0
    with utils.open(input_file_path, 'rb') as file:
        for line in file:
            article = json.loads(line)
            title = article["title"]
            text = article["text"]
            # print(article.keys())
            doc = TaggedDocument(utils.simple_preprocess(text), [title])
            # print(doc)
            # time.sleep(30)
            train_list.append(doc)
            count += 1
            if count % 1000 == 0:
                logger.info("Finished processing %d articles.", count)

    logger.info("Finished processing all of the articles.")

    model = Doc2Vec(train_list, dm=0, dbow_words=1, vector_size=300, window=8, min_count=15, epochs=10, workers=multiprocessing.cpu_count())
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
