from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
            article = json.loads(line)
            train_list.append(TaggedDocument(utils.simple_preprocess(article["text"]), [article["title"]]))
            if count % 10000 == 0:
                logger.info("Finished processing %d articles.", count)

    # with utils.open(input_file_path, 'rb') as file:
    #     for line in file:
    #         count+=1
    #         article = json.loads(line)
    #         text = article['text']
    #         title = article['title']
    #         train_list.append(TaggedDocument(utils.simple_preprocess(doc), [title]))
    #         if count % 10000 == 0:
    #             logger.info("Finished processing %d articles.", count)

    logger.info("Finished processing all of the articles.")
    model = Doc2Vec(dm=0, dbow_words=1, vector_size=300, window=8, min_count=15, epochs=10, workers=multiprocessing.cpu_count())

    model.build_vocab(train_list)
    logger.info("Vocab built.")

    model.train(train_list, total_examples=model.corpus_count, epochs=model.epochs)
    logger.info("Model trained.")

    model.save(output_file_path)

def build_vector_dataset(input_file_path, output_file_path, dataset_file_path):
    count = 0
    with utils.open(input_file_path, 'rb') as file:
        model = Doc2Vec.load(output_file_path)
        outfile = utils.open(dataset_file_path, 'wb')
        for line in file:
            count+=1
            article = json.loads(line)
            title = article["title"]
            data_line = {
                "title" : title,
                "coords" : article["coordinates"],
                "embedding" : model.docvecs[title]
            }
            if count % 10000 == 0:
                logger.info("Finished processing %d articles.", count)

            outfile.write((json.dumps(data_line) + "\n").encode('utf-8'))
    outfile.close()

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    default_workers = max(1, multiprocessing.cpu_count() - 1)
    build_doc2vec("./raw/wikipedia/coord_articles_extraced.bz2", "./raw/wikipedia/eric_doc2vec.model", default_workers)
    build_vector_dataset("./raw/wikipedia/coord_articles_extraced.bz2", "./raw/wikipedia/eric_doc2vec.model", "./raw/wikipedia/vector_dataest.bz2")
    # parser.add_argument('-f', '--file', help='Path to the processed wiki json dump.', required=True)
    # parser.add_argument('-o', '--output', help='Path of the doc2vec model that will be saved.', required=True)
    # parser.add_argument('-d', '--dataset', help='Path of the vector dataset that will be created.', required = True)
    # parser.add_argument('-w', '--workers', help='Number of parallel workers for multicore systems.', default = max(1, multiprocessing.cpu_count() - 1))
    # args = parser.parse_args()
    #
    # logger.info("running %s", " ".join(sys.argv))
    #
    # build_doc2vec(
    #     args.file,
    #     args.output,
    #     args.workers
    # )
    #
    # build_vector_dataset(
    #     args.file,
    #     args.output,
    #     args.dataset
    # )
    #
    # logger.info("finished running %s", sys.argv[0])
