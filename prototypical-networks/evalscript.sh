for way in {5,20}
    do
        for shot in {1,5}
            do
                python scripts/predict/few_shot/run_eval.py --model.model_path identitytransform-1shot-gamma0.95/best_model.pt --data.test_query 15 --data.test_way $way --data.test_shot $shot
            done
    done
