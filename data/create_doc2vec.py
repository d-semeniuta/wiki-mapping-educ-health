from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import gensim.utils
import argparse
import logging
import multiprocessing
import sys

def build_doc2vec(input_file_path, output_file_path, num_workers):
    train_list = []

    count = 0

    with gensim.utils.open(input_file_path, 'rb') as file:
        for line in file:
            count+=1
            file_text = json.loads(line)['text']
            title = json.loads(line)['title']
            train_list.append(TaggedDocument(gensim.utils.simple_preprocess(doc), [title]))
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
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    default_workers = max(1, multiprocessing.cpu_count() - 1)
    parser.add_argument('-f', '--file', help='Path to the processed wiki json dump.', required=True)
    parser.add_argument('-o', '--output', help='Prefix of the doc2vec model that will be saved.', required=True)
    parser.add_argument('-w', '--workers', help='Number of parallel workers for multicore systems.', default = max(1, multiprocessing.cpu_count() - 1))
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))

    build_doc2vec(
        args.file,
        args.output,
        args.workers
    )

    logger.info("finished running %s", sys.argv[0])
