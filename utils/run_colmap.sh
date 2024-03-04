#!/bin/bash

# 输入数据集路径
DATASET_PATH=$1

# 特征提取
colmap feature_extractor \
   --database_path $DATASET_PATH/database.db \
   --image_path $DATASET_PATH/images

# 穷举匹配
colmap exhaustive_matcher \
   --database_path $DATASET_PATH/database.db

# 创建稀疏文件夹
mkdir $DATASET_PATH/sparse

# 映射
colmap mapper \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --output_path $DATASET_PATH/sparse

# 创建密集文件夹
mkdir $DATASET_PATH/dense
