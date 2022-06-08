PROJECT_DIR=${HOME}"/models/t5x_large"
T5X_DIR="../../t5x"  # directory where the t5x is cloned.
MODEL_DIR="gs://myv4-bucket/t5x_large"
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="t5-large.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
