name=VLNBERT-train-Surrogate

flag="--vlnbert prevalent

      --aug data/prevalent/surrogate_data.json
      --test_only 0

      --train surrogate 

      --features places365
      --maxAction 15
      --batchSize 8
      --feedback surrogate
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5
      
      --training_set_custom surrogate10
      --val_set_custom val72
      --load snap/VLNBERT-PREVALENT-final/state_dict/pretrained

      --expert_policy ndtw
      "

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train.py $flag --name $name
