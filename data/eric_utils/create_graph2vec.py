from gensim import utils
import argparse
import logging
import multiprocessing
import sys
import json
from graph2vec.src.graph2vec import create_model

from math import sin, cos, sqrt, atan2, radians
import pandas as pd
from gensim import utils
from gensim.models.doc2vec import Doc2Vec

logger = logging.getLogger(__name__)

def compute_distance(c1, c2):
    # approximate radius of earth in km
    R = 6373.0
    try:
        lat1 = radians(c1[0])
        lon1 = radians(c1[1])
        lat2 = radians(c2[0])
        lon2 = radians(c2[1])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance
    except:
        logger.info("problem with the coordinate - not sure why sad.")
        return -1

def find_clostest_articles(lat, lon, parsed_path, num_nearest=10):
    distances = []
    count = 0
    with utils.open(parsed_path, 'rb') as file:
        for line in file:
            count+=1
            data = json.loads(line)
            curr_dist = compute_distance([lat, lon], data["coords"])
            if curr_dist != -1:
                distances.append((data["title"], data["interlinks"]))
    distances.sort()

    logger.info("Found the " + str(num_nearest) + " nearest articles")
    return [[distances[i][0], distances[i][1]] for i in range(num_nearest)]

def extract_interlink_titles(interlinks):
    return [interlink[0] for interlink in interlinks]

def create_node_class_map_and_index(base_node, title_list, node_class_map, edge_list, title_collector):
    for i in range(len(title_list)):
        curr_title = title_list[i]
        curr_index = len(node_class_map) + i

        node_class_map[curr_index] = curr_title
        edge_list.append([base_node, curr_index])
        title_collector.append(curr_title)
    return (node_class_map, edge_list, title_collector)

def get_interlinks_given_title(parsed_path, title):
    with utils.open(parsed_path, 'rb') as file:
        for line in file:
            data = json.loads(line)
            if data["title"] == title:
                return data["interlinks"]
    return []

def create_graph(parsed_path, output_path, number, file_path, hops = 1):
    model = Doc2Vec.load(doc2vec_path)
    prefix, _ = file_path.split(".")
    graph_prefix = ""
    if "train" in prefix.lower():
        graph_prefix = "train"
    elif "test" in prefix.lower():
        graph_prefix = "test"
    elif "val" in prefix.lower():
        graph_prefix = "val"
    else:
        ValueError("Dataset entered isn't one of train, test, or val.")
    title_collector = []
    for i in range(len(df["Lat"])):
        output_file_path = os.path.join(output_path, graph_prefix + str(i))

        node_class_map = {0: "CLUSTER_NODE_CLASS"}
        edge_list = []

        clust_lat, clust_lon = df["Lat"][i], df["Lon"][i]

        closest_n = find_clostest_articles(clust_lat, clust_lon, parsed_path, number)
        closest_n_titles = [article[0] for article in closest_n]
        closest_n_interlinks = [article[1] for article in closest_n]

        # ADDS INITIAL N CLOSEST TO THE LINK

        node_class_map, edge_list, title_collector = create_node_class_map_and_index(0, closest_n_titles, node_class_map, edge_list, title_collector)

        # GETS ARTICLES 1 HOP AWAY

        for i in range(len(closest_n_interlinks)):
            curr_interlinks = closest_n_interlinks[i]
            curr_interlink_titles = extract_interlink_titles(curr_interlinks)
            node_class_map, edge_list, title_collector = create_node_class_map_and_index(i+1, curr_interlink_titles, node_class_map, edge_list, title_collector)

        # GETS ARTICLES 2 HOPS AWAY

        if hops == 2:
            for i in range(len(closest_n_interlinks)):
                curr_interlinks = closest_n_interlinks[i]
                curr_interlink_titles = extract_interlink_titles(curr_interlinks)
                for title in curr_interlink_titles:
                    base_node = 0
                    for key, value in node_class_map.items():
                        if value == title:
                            base_node = key
                            break

                    # for some reason if there is an error, keep going
                    if base_node == 0:
                        continue

                    interlinks_from_interlink = get_interlinks_given_title(parsed_path, title)
                    if len(interlinks_from_interlink) != 0:
                        new_interlink_titles = extract_interlink_titles(interlinks_from_interlink)
                        node_class_map, edge_list, title_collector = create_node_class_map_and_index(base_node, new_interlink_titles, node_class_map, edge_list, title_collector)

        output = {"edges": edge_list, "features": node_class_map}
        with open(output_file_path, 'w') as outfile:
            json.dump(output, outfile)

    return title_collector


def create_graphs(doc2vec_path, parsed_path, output_folder, feature_file, number, train_path, test_path, valid_path, hops=1):
    titles = create_graph(parsed_path, output_folder, number, train_path, hops)
    titles += create_graph(parsed_path, output_folder, number, test_path, hops)
    titles += create_graph(parsed_path, output_folder, number, valid_path, hops)
    titles = list(set(titles)) # remove duplicates
    with open(feature_file, 'w') as outfile:
        model = Doc2Vec.load(doc2vec_path)
        for title in titles:
            embedding = model.docvecs[title]
            data_string = ",".join(map(str, embedding))
            data_string = title + "," + data_string
            outfile.write(data_string)

def build_graph2vec(graphs_path, output_path, num_workers):
    create_model(graphs_path, output_path, dimensions=256, workers=num_workers)

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
