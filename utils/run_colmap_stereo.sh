#!/bin/bash

# 根据https://github.com/dunbar12138/DSNeRF/issues/102
DATASET_PATH=$1

colmap feature_extractor \
    --database_path $DATASET_PATH/database_stereo_v2.db \
    --image_path $DATASET_PATH/images  \
    --ImageReader.single_camera_per_folder 1 \
    # --SiftExtraction.max_image_size 4032 \
    # --SiftExtraction.max_num_features 32768 \
    # --SiftExtraction.estimate_affine_shape 1 \
    # --SiftExtraction.domain_size_pooling 1

colmap sequential_matcher \
    --database_path $DATASET_PATH/database_stereo_v2.db \
    --SequentialMatching.loop_detection 1 \

mkdir $DATASET_PATH/sparase_stereo_v2

colmap mapper \
    --database $DATASET_PATH/database_stereo_v2.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparase_stereo_v2 \
    # --Mapper.ba_local_max_num_iterations 40 \
    # --Mapper.ba_local_max_refinements 3 \
    # --Mapper.ba_global_max_num_iterations 100