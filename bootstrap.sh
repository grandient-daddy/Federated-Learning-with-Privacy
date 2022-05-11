#!/bin/bash

root_dir=$(dirname "$0")
data_dir="$root_dir/data"
echo "Repository root directory is $root_dir"
echo "Repository data directory is $data_dir"

paths=("metrics_results/lsapp_score_params"
       "metrics_results/lsapp"
       "dynamic_results/lsapp/pictures"
       "dynamic_results/dataframes")

for path in ${paths[@]}; do
    mkdir -pv "$root_dir/$path"
done

wget -c -q --show-progress -P "$data_dir" \
    "https://github.com/aliannejadi/LSApp/raw/main/lsapp.tsv.gz"

tar xfC $data_dir/lsapp.tsv.gz $data_dir
