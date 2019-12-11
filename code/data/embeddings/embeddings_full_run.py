from get_wiki_data import download_wikipedia
from parse_wiki import segment_and_write_all_articles
from create_doc2vec import build_doc2vec
from create_doc2vec_dataset import main as doc2vec_dataset_main
from create_graph2vec import main as graph2vec_dataset_main
from graph2vec.src.graph2vec import create_model
import os

import logging
logger = logging.getLogger(__name__)

import multiprocessing

DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() - 1)

def main():
    download_wikipedia(
        destination = "../raw/wikipedia",
        url = 'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
    )

    segment_and_write_all_articles(
        file_path = "../raw/wikipedia/full_wiki_xml_corpus.bz2",
        output_file = "../raw/wikipedia/processed_wiki_corpus_coord_articles_only.bz2",
        min_article_character = 200,
        workers = DEFAULT_WORKERS,
        include_interlinks=True,
        coords_only=True
    )

    segment_and_write_all_articles(
        file_path = "../raw/wikipedia/full_wiki_xml_corpus.bz2",
        output_file = "../raw/wikipedia/processed_wiki_corpus_all_articles.bz2",
        min_article_character = 200,
        workers = DEFAULT_WORKERS,
        include_interlinks = True,
        coords_only = False
    )

    build_doc2vec(
        input_file_path = "../raw/wikipedia/processed_wiki_corpus_coord_articles_only.bz2",
        output_file_path = "../models/coord_articles_only_doc2vec.model",
        num_workers = DEFAULT_WORKERS
    )

    build_doc2vec(
        input_file_path = "../raw/wikipedia/processed_wiki_corpus_all_articles.bz2",
        output_file_path = "../models/all_articles_doc2vec.model",
        num_workers = DEFAULT_WORKERS
    )

    doc2vec_dataset_main()

    graph2vec_dataset_main()

    create_model("./graphs/two_hop_run2", "../processed/two_hop.csv", dimensions = 300)

if __name__ == "__main__":
    main()
