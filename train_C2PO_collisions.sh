#! /bin/sh

python C2PO_trainer.py --gpus 1 2 3 4  --beta 5.0  --new_first_action_inf True --ad_collisions True --trans_pred True \
    --logdir ./C2PO_collision_logs/  --random_seed 1238 --sigma_chi 0.1