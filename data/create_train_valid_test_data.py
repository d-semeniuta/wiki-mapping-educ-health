from math import sin, cos, sqrt, atan2, radians
from gensim import utils
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

def find_clostest_articles(lat, lon, vector_dataset_path, num_nearest=10):
    distances = []
    count = 0
    with utils.open(vector_dataset_path, 'rb') as vector_file:
        for line in vector_file:
            count+=1
            data = json.loads(line)
            curr_dist = compute_distance([lat, lon], data["coords"])
            if curr_dist != -1:
                distances.append((curr_dist, data["title"], data["coords"], data["embedding"]))
            # logger.info("Finished processing %d articles.", count)
    distances.sort()

    # Get all articles within MARGIN
    # ind = 0
    # while ind < len(distances) and distances[ind][0] <= MARGIN:
    #     ind+=1
    # if ind < len(distances):
    #     # Distance, title, coordinates, embedding
    #     within_m = [[distances[i][0], distances[i][1], distances[i][2], distances[i][3]] for i in range(ind)]
    # else:
    #     within_m = []
    # logger.info("Found " +str(ind) + " articles within " + MARGIN+ "km of " + str(lat) + ", " + str(lon))

    # Get 10 nearest articles
    # Distance, title, coordinates, embedding
    # n_10 = [[distances[i][0], distances[i][1], distances[i][2], distances[i][3]] for i in range(num_nearest)]
    n_10 = [[distances[i][0], distances[i][3]] for i in range(num_nearest)]
    logger.info("Found the " + str(num_nearest) + " nearest articles")
    # return [within_m, n_10]
    return n_10

def create_ez_dataset(vector_path, file_path):
    prefix, _ = file_path.split(".")
    outfile = utils.open(prefix + ".bz2", 'wb')
    count = 0
    logger.info("Beginning to ez create dataset for " + file_path)
    with open(file_path, 'rb') as train_data:
        first_pass = True
        for line in train_data:
            if first_pass:
                continue
            else:
                count+=1
                split_data = line.split(",")
                output_data = find_clostest_articles(split_data[6], split_data[7], vector_path)
                useful_data = split_data[8:13]
                output_string = split_data[3] + "," + ",".join(useful_data)
                dist_list = []
                embedding_list = []
                for i in range(10):
                    dist_list.append(output_data[i][0])
                    embedding_list += output_data[i][1]
                output_string += "," + ",".join(embedding_list) + "," + ",".join(dist_list)
                if count % 1000 == 0:
                    logger.info("Finished processing %d datapoints.", count)
                outfile.write((output_string + "\n").encode('utf-8'))


def create_ez_datasets(vector_path, train_path, test_path, valid_path):
    create_ez_dataset(vector_path, train_path)
    create_ez_dataset(vector_path, test_path)
    create_ez_dataset(vector_path, valid_path)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-v', '--vector', help="Path to the processed vector json dump.", required=True)
    parser.add_argument('--train', help="Path to the train cluster set.", required=True)
    parser.add_argument('--test', help = "Path to the test cluster set.", required=True)
    parser.add_argument('--valid', help = "Path to the validation cluster set.", required=True)
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))

    create_ez_datasets(
        args.vector,
        args.train,
        args.test,
        args.valid
    )

    logger.info("finished running %s", sys.argv[0])
