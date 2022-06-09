PROJECT_DIR=${HOME}"/models/pk-nb-t5x"
T5X_DIR="../../t5x"  # directory where the t5x is cloned.
MODEL_DIR="gs://t5x-backup/norwegian_NCC_plus_English_pluss200k_balanced_bokmaal_nynorsk_pluss100k_lm_t5x_large/"
export PYTHONPATH=${PROJECT_DIR}

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="balanced_bokmaal_nynorsk_lm_large.gin" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \