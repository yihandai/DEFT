name=VLNBERT-test-Prevalent

flag="--vlnbert prevalent

      --submit 0
      --test_only 0

      --train testmask
      --load snap/VLNBERT-PREVALENT-final/state_dict/pretrained
      --loadmask snap/VLNBERT-train-Prevalent/state_dict_mask_num_reward_fix_nondtw/LAST_iter7000

      --features places365
      --maxAction 15
      --batchSize 1
      --feedback sample
      --lr 1e-5
      --iters 300000
      --optim adamW

      --mlWeight 0.20
      --maxInput 80
      --angleFeatSize 128
      --featdropout 0.4
      --dropout 0.5"

mkdir -p snap/$name
CUDA_VISIBLE_DEVICES=0 python -u r2r_src/train.py $flag --name $name
