import pandas as pd
import re
import requests
import datetime
import mmap

import os

data_dir = os.path.abspath('./raw')
processed_dir = os.path.abspath('./processed')
imr_5yr_loc = os.path.join(data_dir, 'InfantMortality_Cluster5Year.csv')
imr_1yr_loc = os.path.join(data_dir, 'InfantMortality_ClusterYear.csv')
mat_ed_loc = os.path.join(data_dir, 'MaternalEducation_cluster.csv')

COORD_ART_PREFIX = "coordinate_articles_"
COORD_ART_SUFFIX = ".txt"

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

# ADD GSCOORD KEY-VALUE IN THE ACTUAL FOR LOOP THROUGH THE GEOCOORDINATES

def build_csv(lat_lon_df, gsradius = 10000, gslimit = 10):
    '''
    lat_lon_df: dataframe of latitudes and longitudes
    gsradius: search radius in meters, int
    gslimit: max number of pages to return, int
    '''
    PARAMS = {
        "format": "json",
        "list": "geosearch",
        "action": "query"
    }
    PARAMS["gslimit"]: gslimit
    PARAMS["gsradius"]: gsradius

    columns = ["lat", "lon"]
    within_distance_string = "Num Articles Within " + gsradius + " Meters of Loc"
    avg_word_count_string = "Avg Word Count"
    avg_rev_count_string = "Avg Revision Count"
    avg_time_since_last_rev_string = "Avg Time Since Last Revision"
    educ_article_count_string = "Educ Article Count"
    health_article_count_string = "Health Article Count"
    columns.append(within_distance_string)
    columns.append(avg_word_count_string)
    columns.append(avg_rev_count_string)
    columns.append(avg_time_since_last_rev_string)
    columns.append(educ_article_count_string)
    columns.append(health_article_count_string)

    compiled_csv = pd.DataFrame(columns=columns)

    for index, row in lat_lon_df.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        PARAMS["gscoord"] = str(lat) + "|" + str(lon)
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()

        PLACES = DATA['query']['geosearch']
        PAGES = []
        PAGE_TITLES = []

        for place in PLACES:
            PAGES.append(place["pageid"])
            PAGE_TITLES.append(place["title"])

        avg_word_count = find_avg_word_counts(PAGES)
        avg_rev_count = find_avg_revisions(PAGES)
        avg_time_since_last_rev = find_avg_time_since_last_rev(PAGES)
        educ_article_count = education_category_count(PAGE_TITLES)
        health_article_count = health_category_count(PAGE_TITLES)
        compiled_csv = compiled_csv.append({
            "lat": lat,
            "lon": lon,
            within_distance_string: len(PLACES),
            avg_word_count_string: avg_word_count,
            avg_rev_count_string: avg_rev_count,
            avg_time_since_last_rev_string: avg_time_since_last_rev,
            educ_article_count_string: educ_article_count,
            health_article_count_string: health_article_count,
        })

    lat_lon_df.to_csv(os.path.join(processed_dir, 'geolocation_stats.csv'), index=False)

def find_avg_word_counts(page_id_list):
    pageids = "|".join(page_id_list)
    PARAMS = {
        "format": "json",
        "prop": "extracts",
        "action": "query",
        "pageids": pageids,
        "explaintext": "true"
    }
    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    PAGES = DATA['query']['pages']
    counts = []
    for page in PAGES:
        line = page["extract"]
        count = len(re.findall(r'\w+', line))
        counts.append(count)
    return sum(counts)/len(counts)

def find_avg_revisions(page_id_list):
    pageids = "|".join(page_id_list)
    PARAMS = {
        "format": "json",
        "prop": "revisions",
        "continue": "",
        "action": "query",
        "pageids": pageids,
        "rvprop": "ids|userid",
        "rvlimit": "max"
    }

    wp_call = requests.get(BASE_URL, params=parameters)
    response = wp_call.json()

    total_revisions = 0

    while True:
      wp_call = requests.get(BASE_URL, params=parameters)
      response = wp_call.json()
      for page_id in response['query']['pages']:
        total_revisions += len(response['query']['pages'][page_id]['revisions'])
      if 'continue' in response:
        parameters['continue'] = response['continue']['continue']
        parameters['rvcontinue'] = response['continue']['rvcontinue']
      else:
        break
    return (total_revisions/(len(page_id_list)))

def education_category_count(page_titles_list):
    count = 0
    educ_file_list = ["college", "institute", "library", "school", "university"]
    for title in page_titles_list:
        for file in educ_file_list:
            path = os.path.join(data_dir, COORD_ART_PREFIX + file + COORD_ART_SUFFIX)
            if find_string_in_file(path, title):
                count += 1
                break
    return count

def health_category_count(page_titles_list):
    count = 0
    health_file_list = ["hospital"]
    for title in page_titles_list:
        for file in educ_file_list:
            path = os.path.join(data_dir, COORD_ART_PREFIX + file + COORD_ART_SUFFIX)
            if find_string_in_file(path, title):
                count += 1
                break
    return count

def find_string_in_file(path, target):
    with open(path, 'rb', 0) as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        if s.find(str.encode(target)) != -1:
            return True

def find_avg_time_since_last_rev(page_id_list):
    pageids = "|".join(page_id_list)
    PARAMS = {
        "action": "query",
        "prop": "revisions",
        "pageids": pageids,
        "rvprop": "timestamp",
        "rvslots": "main",
        "formatversion": "2",
        "format": "json"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()

    PAGES = DATA['query']['pages']
    timestamps = []
    for page in pages:
        timestamp = page["timestamp"]
        dt = datetime.fromisoformat(timestamp)
        timestamps.append(unix_time_millis(dt))

    return sum(timestamps)/len(timestamps)

def unix_time_millis(dt):
    epoch = datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000.0

def main():
    imr_1yr = pd.read_csv(imr_1yr_loc)
    imr_5yr = pd.read_csv(imr_5yr_loc)
    mat_ed = pd.read_csv(mat_ed_loc)

    lat_lon_df = imr_1yr[['lat', 'lon']]
    lat_lon_df = lat_lon_df.append(imr_5yr[['lat', 'lon']], ignore_index=True)
    lat_lon_df = lat_lon_df.append(mat_ed[['lat', 'lon']], ignore_index=True)

    lat_lon_df.drop_duplicates(keep='first', inplace=True)
    build_csv(lat_lon_df)

if __name__ == '__main__':
    main()
