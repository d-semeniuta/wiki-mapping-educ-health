import pprint
pp = pprint.PrettyPrinter(indent=4)
import os
import multiprocessing
​
import pickle
import pandas as pd
​
# from multiprocessing import Pool
# import multiprocessing
​
import logging
from gensim import utils
import json
import networkx as nx
​
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
​
df = pd.read_csv('../split/nearest_articles_2.csv', sep=';', header = None)
df_cols = df.columns
max_len = len(df[df_cols[0]])
​
curr_path = os.path.abspath(os.curdir)
​
with open(os.path.join(curr_path, "four_country_three_hop_dict.pkl"), 'rb') as fp:
    total_interlinks_dict = pickle.load(fp)
​
test_dict = dict(total_interlinks_dict)
​
###################### GRAPH STATISTICS ####################################
one_hop_stats = {
    "node_degrees": [],
    # "average_degree": [],
    # "average_clustering_coeff": [],
    # "degree_centrality": [],
    # "diameter": []
}
​
two_hop_stats = {
    "node_degrees": [],
    # "average_degree": [],
    # "average_clustering_coeff": [],
    # "degree_centrality": [],
    # "diameter": []
}
​
# three_hop_stats = {
#     "node_degrees": [],
#     "average_degree": [],
#     "average_clustering_coeff": [],
#     "degree_centrality": [],
#     "diameter": []
# }
​
def create_graph_by_cluster(row_index):
​
    row = list(df.loc[row_index].to_numpy())
    cluster_id = row[0] # check
    titles = row[1:11] # check
​
    # ONE HOP CONSTRUCTION
​
    outfile_path_one_hop = os.path.join(curr_path, "graphs", "one_hop_four_country", "{}.json".format(str(cluster_id)))
​
    graph = {"BASE NODE": titles} # check
    for title in titles:
        graph[title] = test_dict[title] # check
​
    number_graph = create_number_ordering_of_graphs(graph)
    edge_list = create_edge_list(number_graph)
​
    G = nx.Graph()
    G.add_edges_from(edge_list)
​
    graph_degrees = [val for (node, val) in G.degree()]
    one_hop_stats['node_degrees'] += [val for (node, val) in G.degree()]
    # one_hop_stats['average_degree'].append( float(sum(graph_degrees)) / len(graph_degrees) )
    # one_hop_stats['average_clustering_coeff'].append(nx.average_clustering(G))
    # one_hop_stats['degree_centrality'].append(nx.degree_centrality(G))
    # one_hop_stats['diameter'].append(nx.diameter(G))
​
    output = {"edges": edge_list}
    with open(outfile_path_one_hop, 'w') as file:
        json.dump(output, file)
​
    # TWO HOP CONSTRUCTION
​
    outfile_path_two_hop = os.path.join(curr_path, "graphs", "two_hop_four_country", "{}.json".format(str(cluster_id)))
​
    existing_keys = graph.keys()
​
    all_titles = list(existing_keys)
    for key_title in existing_keys:
        all_titles += graph[key_title]
​
    all_titles = set(all_titles)
    for title in all_titles:
        if title not in existing_keys:
            try:
                graph[title] = test_dict[title]
            except:
                pass
​
    number_graph = create_number_ordering_of_graphs(graph)
    edge_list = create_edge_list(number_graph)
​
    G = nx.Graph()
    G.add_edges_from(edge_list)
​
    graph_degrees = [val for (node, val) in G.degree()]
    two_hop_stats['node_degrees'] += [val for (node, val) in G.degree()]
    # two_hop_stats['average_degree'].append( float(sum(graph_degrees)) / len(graph_degrees) )
    # two_hop_stats['average_clustering_coeff'].append(nx.average_clustering(G))
    # two_hop_stats['degree_centrality'].append(nx.degree_centrality(G))
    # two_hop_stats['diameter'].append(nx.diameter(G))
​
    output = {"edges": edge_list}
​
    with open(outfile_path_two_hop, 'w') as file:
        json.dump(output, file)
​
def create_graph_by_cluster_parallel(in_queue, out_dict_one, out_dict_two):
​
    row = list(df.loc[row_index].to_numpy())
    cluster_id = row[0] # check
    titles = row[1:11] # check
​
    # ONE HOP CONSTRUCTION
​
    outfile_path_one_hop = os.path.join(curr_path, "graphs", "one_hop_run2", "{}.json".format(str(cluster_id)))
​
    graph = {"BASE NODE": titles} # check
    for title in titles:
        graph[title] = test_dict[title] # check
​
    number_graph = create_number_ordering_of_graphs(graph)
    edge_list = create_edge_list(number_graph)
​
    G = nx.Graph()
    G.add_edges_from(edge_list)
​
    graph_degrees = [val for (node, val) in G.degree()]
    one_hop_stats['node_degrees'] += [val for (node, val) in G.degree()]
    one_hop_stats['average_degree'].append( float(sum(graph_degrees)) / len(graph_degrees) )
    one_hop_stats['average_clustering_coeff'].append(nx.average_clustering(G))
    one_hop_stats['degree_centrality'].append(nx.degree_centrality(G))
    one_hop_stats['diameter'].append(nx.diameter(G))
​
    output = {"edges": edge_list}
    with open(outfile_path_one_hop, 'w') as file:
        json.dump(output, file)
​
    # TWO HOP CONSTRUCTION
​
    outfile_path_two_hop = os.path.join(curr_path, "graphs", "two_hop_run2", "{}.json".format(str(cluster_id)))
​
    existing_keys = graph.keys()
​
    all_titles = list(existing_keys)
    for key_title in existing_keys:
        all_titles += graph[key_title]
​
    all_titles = set(all_titles)
    for title in all_titles:
        if title not in existing_keys:
            try:
                graph[title] = test_dict[title]
            except:
                pass
​
    number_graph = create_number_ordering_of_graphs(graph)
    edge_list = create_edge_list(number_graph)
​
    G = nx.Graph()
    G.add_edges_from(edge_list)
​
    graph_degrees = [val for (node, val) in G.degree()]
    two_hop_stats['node_degrees'] += [val for (node, val) in G.degree()]
    two_hop_stats['average_degree'].append( float(sum(graph_degrees)) / len(graph_degrees) )
    two_hop_stats['average_clustering_coeff'].append(nx.average_clustering(G))
    two_hop_stats['degree_centrality'].append(nx.degree_centrality(G))
    two_hop_stats['diameter'].append(nx.diameter(G))
​
    output = {"edges": edge_list}
​
    with open(outfile_path_two_hop, 'w') as file:
        json.dump(output, file)
​
​
def create_number_ordering_of_graphs(graph):
    total_list = []
    for key, values in graph.items():
        total_list.append(key)
        total_list += values
​
    number_mapping = {}
    for item in total_list:
        number_mapping[item] = len(number_mapping)
​
    new_number_graph = {}
    for key, values in graph.items():
        new_key_marker = number_mapping[key]
        new_val_list = [number_mapping[val] for val in values]
        new_number_graph[new_key_marker] = new_val_list
​
    return new_number_graph
​
def create_edge_list(graph):
    edges = []
​
    for node in graph.keys():
        for neighbor in graph[node]:
            edge = [node, neighbor]
            same_edge = [neighbor, node]
            if same_edge not in edges and edge not in edges:
                edges.append(edge)
​
    return edges
​
def main():
    for i in range(max_len):
        create_graph_by_cluster(i)
        logger.info("Gotten through {}".format(str(i)))
​
​
# NUM_WORKERS = multiprocessing.cpu_count() - 1
#
# manager = Manager()
# results = manager.dict()
# work = manager.Queue(NUM_WORKERS)
# for i in range(NUM_WORKERS):
#     p = Process(target = )
​
with open(os.path.join(curr_path, "one_hop_stats_dict_four_country.pkl"), 'wb') as fp:
    pickle.dump(one_hop_stats, fp)
​
with open(os.path.join(curr_path, "two_hop_stats_dict_four_country.pkl"), 'wb') as fp:
    pickle.dump(two_hop_stats, fp)
