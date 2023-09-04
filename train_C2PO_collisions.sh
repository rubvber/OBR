#! /bin/sh

python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8 --batch_size 8 --val_batch_size 16  --beta 5.0  --new_first_action_inf True --ad_collisions True --trans_pred True \
    --logdir ./C2PO_collision_logs/  --random_seed 1238  