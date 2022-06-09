PROJECT_DIR=${HOME}"/models/t5x_models"
T5X_DIR="../../t5x"  # directory where the t5x is cloned.
MODEL_DIR="gs://t5x-training/pretrained_models/test"
MODEL_DIR="gs://myv4-bucket/mt5x_large_temp"
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="ro_mt5_large.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
