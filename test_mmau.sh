#!/bin/bash

TEST_FILE=data/MMAU/mmau-test-mini.json
model_path=$1
iters=($2)
for iter in ${iters[*]}; do
    MODEL_DIR=$1/checkpoint-${iter}
    OUT_DIR=$1/test_${iter}
    mkdir -p ${OUT_DIR} || exit 1
    ln -s "$(pwd)/exp/spk_dict.pt" "${MODEL_DIR}/spk_dict.pt"
    python src/test.py --model_path ${MODEL_DIR} --batch_size 8 --data_file ${TEST_FILE} --out_file ${OUT_DIR}/res_mmau.json --think True --think_max_len 50 || exit 1
    python data/MMAU/evaluation.py --input ${OUT_DIR}/res_mmau.json > ${OUT_DIR}/eval_mmau.json || exit 1

    echo "Completed checkpoint ${iter}. Generated done file: ${OUT_DIR}/test_${iter}.done"
done

# show Acc for each checkpoint
python src/utils/show_acc.py -i $1 || exit 1
