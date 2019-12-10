from math import sin, cos, sqrt, atan2, radians
import pandas as pd
from gensim import utils
from gensim.models.doc2vec import Doc2Vec
import json
import logging
import argparse
import sys
import multiprocessing
from multiprocessing import Process, Queue
import pickle
​
logger = logging.getLogger(__name__)
​
# Distance in km to check within
MARGIN = 10
​
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
​
def find_clostest_articles(lat, lon, parsed_array, num_nearest=10):
    distances = []
    # count = 0
    for entry in parsed_array:
        # count+=1
        curr_dist = compute_distance([lat, lon], entry[0])
        if curr_dist != -1:
            distances.append([curr_dist, entry[1]])
        # if count % 10000 == 0:
        #     logger.info("Computed {} distances".format(str(count)))
            # logger.info("Finished processing %d articles.", count)
    distances.sort()
​
    # logger.info("Found the " + str(num_nearest) + " nearest articles")
    return [[distances[i][0], distances[i][1]] for i in range(num_nearest)]
​
def line_writer(i):
    # print('Doing index {}'.format(str(i)))
​
    df = pd.read_csv("../split/ClusterLevelCombined_5yrIMR_MatEd.csv", header = 0)
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    df = df[df['country'].isin(countries)]
    # output_file_path = "./split/ClusterLevelCombined_5yrIMR_MatEd_train_dataset.csv"
    # with open(output_file_path, 'w') as outfile:
    clust_lat, clust_lon = df['lat'].iloc[i], df['lon'].iloc[i]
    dist_array = find_clostest_articles(clust_lat, clust_lon, parsed_array)
    output_string = str(df['id'].iloc[i]) + ";"
    embedding_strings = []
    distances = []
    for entry in dist_array:
        distances.append(str(entry[0]))
        output_string += entry[1] + ";"
    output_string += ";".join(distances)
    return output_string
​
def main():
    p = multiprocessing.Pool()
    df = pd.read_csv("../split/ClusterLevelCombined_5yrIMR_MatEd.csv", header = 0)
    countries = ['Ghana', 'Zimbabwe', 'Kenya', 'Egypt']
    df = df[df['country'].isin(countries)]
​
    print(df.country.unique())
​
    max_len = len(df['lat'])
    logger.info("Need to process {} datapoints".format(max_len))
    count = 0
    with open('../split/nearest_articles_2.csv', 'w') as outfile:
        for result in p.imap(line_writer, range(max_len)):
            outfile.write(result + "\n")
            count += 1
            if count % 100 == 0:
                logger.info("Processed {} lines".format(str(count)))
​
​
if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    main()
