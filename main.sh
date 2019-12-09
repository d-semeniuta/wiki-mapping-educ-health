#!bin/bash

CODE_HEAD="$(pwd)/code"

# download necessary data
cd "$CODE_HEAD/data"
python3 get_wiki_data.py

# run data processing
mkdir processed
cd "$CODE_HEAD/data/embeddings"
python3 embeddings_full_run.py

cd "$CODE_HEAD/util"
python3 make_image_set.py

# run model
cd $CODE_HEAD
python3 train.py
