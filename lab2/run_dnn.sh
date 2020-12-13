#!/usr/bin/env bash

DATA_PATH=./data/test


# FIXME: CHANGE THESE PATHS TO MATCH YOUR CONFIG
GRAPH_PATH=./exp_tri1/graph
TEST_ALI_PATH=./exp_tri1_ali_test
OUT_DECODE_PATH=./exp_tri1/decode_test_dnn


CHECKPOINT_FILE=./best_usc_dnn.pt
DNN_OUT_FOLDER=./dnn_out

# ------------------- Data preparation for DNN -------------------- #
# Compute cmvn stats for every set and save them in specific .ark files
# These will be used by the python dataset class that you were given
for set in train dev test; do
  compute-cmvn-stats --spk2utt=ark:data/${set}/spk2utt scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_speaker.ark"
  compute-cmvn-stats scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_snt.ark"
done


# ------------------ TRAIN DNN ------------------------------------ #
python timit_dnn.py $CHECKPOINT_FILE


# ----------------- EXTRACT DNN POSTERIORS ------------------------ #
python extract_posteriors $CHECKPOINT_FILE $DNN_OUT_FOLDER


# ----------------- RUN DNN DECODING ------------------------------ #
./decode_dnn.sh $GRAPH_PATH $DATA_PATH $TEST_ALI_PATH $OUT_DECODE_PATH "cat $DNN_OUT_FOLDER/posteriors.ark"
