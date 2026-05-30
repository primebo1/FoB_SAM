#!/bin/bash
# test a model to segment abdominal/cardiac MRI
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='isic'

NWORKER=16
RUNS=1
ALL_EV=(1 2 3) # 3 dicease classes (1, 2, 3)
TEST_LABEL=[]
###### Training configs ######
NSTEP=39001
DECAY=0.98

MAX_ITER=3000 # defines the size of an epoch
SNAPSHOT_INTERVAL=3000 # interval for saving snapshot
SEED=2025

N_PART=3 # defines the number of chunks for evaluation
ALL_SUPP=(2) 
echo ========================================================================
for EVAL_FOLD in "${ALL_EV[@]}"
do
    PREFIX="test_${DATASET}_cv${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./results"

    if [ ! -d $LOGDIR ]
    then
      mkdir -p $LOGDIR
    fi
    for SUPP_IDX in "${ALL_SUPP[@]}"
    do
      # RELOAD_PATH='please feed the absolute path to the trained weights here' # path to the reloaded model
      RELOAD_MODEL_PATH=".../exps_train_on_isic_FSMIS_FoB/FSMIS_train_isic_cv${EVAL_FOLD}/1/snapshots/39000.pth"
      python3 test.py with \
      mode="test" \
      dataset=$DATASET \
      num_workers=$NWORKER \
      n_steps=$NSTEP \
      eval_fold=$EVAL_FOLD \
      max_iters_per_load=$MAX_ITER \
      supp_idx=$SUPP_IDX \
      test_label=$TEST_LABEL \
      seed=$SEED \
      n_part=$N_PART \
      reload_model_path=$RELOAD_MODEL_PATH \
      save_snapshot_every=$SNAPSHOT_INTERVAL \
      lr_step_gamma=$DECAY \
      path.log_dir=$LOGDIR
  done
done






