from math import sin, cos, sqrt, atan2, radians
import pandas as pd
from gensim import utils
from gensim.models.doc2vec import Doc2Vec
import json
import logging

logger = logging.getLogger(__name__)

# Distance in km to check within
MARGIN = 10

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
                distances.append((curr_dist, data["title"]))
            # logger.info("Finished processing %d articles.", count)
    distances.sort()

    logger.info("Found the " + str(num_nearest) + " nearest articles")
    return [[distances[i][0], distances[i][1]] for i in range(num_nearest)]

def create_doc2vec_dataset(doc2vec_path, parsed_path, number, file_path):
    model = Doc2Vec.load(doc2vec_path)
    prefix, _ = file_path.split(".")
    outfile_path = prefix + "_doc2vec_dataset.csv"
    with open(output_file_path, 'wb') as outfile:
        df = pd.read_csv(file_path, header = True)
        for i in range(len(df["Lat"])):
            embedding_collector = []
            distance_collector = []
            clust_lat, clust_lon = df["Lat"][i], df["Lon"][i]
            topn = find_clostest_articles(clust_lat, clust_lon, parsed_path, number)
            for entry in topn:
                title = entry[1]
                distance = entry[0]
                distance_collector.append(distance)
                embedding = model.docvecs[title]
                data_string = ",".join(map(str, embedding))
                embedding_collector.append(data_string)
            final_collector = embedding_collector + distance_collector
            output_line = ",".join(map(str, final_collector))
            # CAN ADD THE Y VALUES IN HERE TOO.
            outfile.write(output_line + "\n")

def create_doc2vec_datasets(doc2vec_path, parsed_path, number, train_path, test_path, valid_path):
    create_doc2vec_dataset(doc2vec_path, parsed_path, number, train_path)
    create_doc2vec_dataset(doc2vec_path, parsed_path, number, test_path)
    create_doc2vec_dataset(doc2vec_path, parsed_path, number, valid_path)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-d', '--doc2vec', help="Path to the doc2vec modelS.", required=True)
    parser.add_argument('-p', '--parsed', help="Path to the parsed dataset.", required=True)
    parser.add_argument('-n', '--number', help="Number of closest articles to find", default=10)
    parser.add_argument('--train', help="Path to the train cluster set.", required=True)
    parser.add_argument('--test', help = "Path to the test cluster set.", required=True)
    parser.add_argument('--valid', help = "Path to the validation cluster set.", required=True)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))

    create_doc2vec_datasets(
        args.doc2vec,
        args.parsed,
        args.number,
        args.train,
        args.test,
        args.valid
    )

    logger.info("finished running %s", sys.argv[0])
