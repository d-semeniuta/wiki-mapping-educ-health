from eric_utils.get_wiki_data import download_wikipedia
from eric_utils.parse_wiki import segment_and_write_all_articles
from eric_utils.create_doc2vec import build_doc2vec
from eric_utils.create_dataset import create_doc2vec_datasets
from eric_utils.create_graph2vec import create_graphs

import logging
logger = logging.getLogger(__name__)

import multiprocessing

DEFAULT_WORKERS = max(1, multiprocessing.cpu_count() - 1)

def main():
    download_wikipedia(
        destination = "./raw/wikipedia"
    )

    segment_and_write_all_articles(
        file_path = "./raw/wikipedia/full_wiki_xml_corpus.bz2",
        output_file = "./raw/wikipedia/processed_wiki_corpus_coord_articles_only.bz2",
        min_article_character = 200,
        workers = DEFAULT_WORKERS,
        include_interlinks=True,
        coords_only=True
    )

    segment_and_write_all_articles(
        file_path = "./raw/wikipedia/full_wiki_xml_corpus.bz2",
        output_file = "./raw/wikipedia/processed_wiki_corpus_all_articles.bz2",
        min_article_character = 200,
        workers = DEFAULT_WORKERS,
        include_interlinks = True,
        coords_only = False
    )

    build_doc2vec(
        input_file_path = "./raw/wikipedia/processed_wiki_corpus_coord_articles_only.bz2",
        output_file_path = "./models/coord_articles_only_doc2vec.model",
        num_workers = DEFAULT_WORKERS
    )

    build_doc2vec(
        input_file_path = "./raw/wikipedia/processed_wiki_corpus_all_articles.bz2",
        output_file_path = "./models/all_articles_doc2vec.model",
        num_workers = DEFAULT_WORKERS
    )

    create_doc2vec_dataset(
        doc2vec_path = "./models/coord_articles_only_doc2vec.model",
        parsed_path = "./raw/wikipedia/processed_wiki_corpus_coord_articles_only.bz2",
        number = 10,
        train_path = "./split/ClusterLevelCombined_5yrIMR_MatEd_train.csv",
        test_path = "./split/ClusterLevelCombined_5yrIMR_MatEd_test.csv",
        valid_path = "./split/ClusterLevelCombined_5yrIMR_MatEd_valid.csv"
    )

    create_graphs(
        doc2vec_path = "./models/all_articles_doc2vec.model",
        parsed_path = "./raw/wikipedia/processed_wiki_corpus_all_articles.bz2",
        output_folder = "./graph2vec/eric_dataset",
        feature_file = "./graph2vec/eric_features/feat.csv",
        number = 10,
        train_path = "./split/ClusterLevelCombined_5yrIMR_MatEd_train.csv",
        test_path = "./split/ClusterLevelCombined_5yrIMR_MatEd_test.csv",
        valid_path = "./split/ClusterLevelCombined_5yrIMR_MatEd_valid.csv",
        hops = 2
    )

    build_graph2vec(
        graphs_path = "./graph2vec/eric_dataset",
        output_path = "./models/graph2vec.csv",
        num_workers = DEFAULT_WORKERS
    )

if __name__ == "__main__":
    main()
