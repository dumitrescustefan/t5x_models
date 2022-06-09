PROJECT_DIR=${HOME}"/models/pk-nb-t5x"
T5X_DIR="../../t5x"  # directory where the t5x is cloned.
MODEL_DIR="gs://t5x-training/pretrained_models/scandinavian_t5x_small"
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="scandinavian_solo_small.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
