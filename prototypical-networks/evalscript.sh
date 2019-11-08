#!/bin/bash
name=${1-omniglotbaseline}
echo $name

for way in {5,20}
    do
        for shot in {1,}
            do
                python scripts/predict/few_shot/run_eval.py \
                --model.model_path results/$name/best_model.pt --data.test_query 15 --data.test_way $way --data.test_shot $shot --augment_stn 0
            done
    done
