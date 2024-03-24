#!/bin/bash

# 根据https://github.com/dunbar12138/DSNeRF/issues/102
DATASET_PATH=$1
SPARSE_PATH=$2

colmap feature_extractor \
    --database_path $DATASET_PATH/database_official.db \
    --image_path $DATASET_PATH/images  \
    --SiftExtraction.max_image_size 4032 \
    --SiftExtraction.max_num_features 32768 \
    --SiftExtraction.estimate_affine_shape 1 \
    --SiftExtraction.domain_size_pooling 1

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database_official.db \
    --SiftMatching.guided_matching 1 \
    --SiftMatching.max_num_matches 65536

mkdir $SPARSE_PATH

colmap mapper \
    --database $DATASET_PATH/database_official.db \
    --image_path $DATASET_PATH/images \
    --output_path $SPARSE_PATH \
    --Mapper.ba_local_max_num_iterations 40 \
    --Mapper.ba_local_max_refinements 3 \
    --Mapper.ba_global_max_num_iterations 100