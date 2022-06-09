PROJECT_DIR=${HOME}"/models/t5x_models"
T5X_DIR="../../t5x"  # directory where the t5x is cloned.
MODEL_DIR="gs://myv4-bucket/ro_mt5x_base"
GIN_FILE="ro_mt5x_base.gin"

export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file=${GIN_FILE} \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
