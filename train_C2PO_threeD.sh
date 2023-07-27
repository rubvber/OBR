#! /bin/sh

# python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  \
# --resume_from_checkpoint /home/rubber/C2PO/C2PO_logs/lightning_logs/version_10/checkpoints/last.ckpt \
# --max_epochs 400
# v1, v2, v3, v8, v9, v10, v11

# python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  --n_latent 32 --learning_rate 1e-4
#Didn't work

# python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  --reduceLR_patience 8 --with_rotation True \
# --resume_from_checkpoint /home/rubber/C2PO/C2PO_logs/lightning_logs/version_6/checkpoints/last.ckpt
#v6 

# python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  --ad_num_frames 1 --ad_val_num_frames 1 \
    # --val_predict 0
#v12

# python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  --ad_num_frames 6 --ad_val_num_frames 10  \
#     --val_batch_size 16 --ad_bounding_actions True --D_init_sd 0.002 --random_seed 1235 --learning_rate 1e-4
#v13: rs 1235, v14: 1234, v15: 1236, v16: 1237 (all lr 3e-4, all failed)
#v17: rs 1235, lr 1e-4

# python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  --ad_bounding_actions True --D_init_sd 0.002 \
#     --train_win_size 3 --val_batch_size 16 --learning_rate 1e-4 --resume_overrideLR True \
#     --resume_from_checkpoint /home/rubber/C2PO/C2PO_logs/lightning_logs/version_18/checkpoints/epoch=28-step=22649.ckpt
#v18, v19 (resumed with lower lr)

# python C2PO_trainer.py --gpus 1 2 3 8  --threeD True --batch_size 16 --reduceLR_factor 0.333333  --ad_bounding_actions True --D_init_sd 0.002 \
    # --val_batch_size 16 --ad_bgcolor 127
#v20

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333  --ad_bounding_actions True --D_init_sd 0.002 \
#     --val_batch_size 16 --ad_bgcolor 127 --beta 5.0 \
#     --resume_from_checkpoint /home/rubber/C2PO/C2PO_logs/lightning_logs/version_21/checkpoints/last.ckpt
#v21: after finding beta bug
#v22: resuming with D logging implemented
#Best one so far

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333  --ad_bounding_actions True --D_init_sd 0.002 \
    # --val_batch_size 16 --beta 5.0 --ad_scale_min 1.5 --ad_scale_max 1.50001
#v23

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333  --ad_bounding_actions True --D_init_sd 0.002 \
#     --val_batch_size 16 --beta 5.0 --ad_scale_min 1.5 --ad_scale_max 1.50001 --network_config greffCLEVR --n_latent 32 --ad_bgcolor 127 \
#     --resume_from_checkpoint /home/rubber/C2PO/C2PO_logs/lightning_logs/version_24/checkpoints/last.ckpt --resume_overrideLR True --learning_rate 1e-4
#v24, v25

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333  --ad_bounding_actions True --D_init_sd 0.002 \
#     --val_batch_size 16 --beta 5.0 --ad_scale_min 1.5 --ad_scale_max 1.50001 --n_latent 32 --ad_bgcolor 127 
#v26

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333  --ad_bounding_actions False --D_init_sd 0.00002 \
#     --val_batch_size 16 --ad_bgcolor 127 --beta 5.0 --action_frames 100 --ad_scale_min 1.5 --ad_scale_max 1.50001 \
#     --resume_from_checkpoint /home/rubber/C2PO/C2PO_logs/lightning_logs/version_28/checkpoints/last.ckpt
#v28: No actions
#v29: v28 resumed

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333 --D_init_sd 0.002 \
    # --val_batch_size 16 --ad_bgcolor 127 --beta 5.0  --ad_scale_min 1.5 --ad_scale_max 1.50001 --sigma_chi 0.1 --init_tau 1.0
#v30: with actions

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333 --D_init_sd 0.002 \
#     --val_batch_size 16 --ad_bgcolor 127 --beta 5.0  --ad_scale_min 1.5 --ad_scale_max 1.50001 --sigma_chi 0.1 --init_tau 1.0 --include_bgd_action False
#v31: with actions, but not on bgd

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333 --reduceLR_patience 8 --D_init_sd 0.002 \
#     --val_batch_size 16 --beta 5.0  --ad_scale_min 1.5 --ad_scale_max 1.50001 --sigma_chi 0.1 --init_tau 1.0
# #v32: with actions, incl bgd, but variable bgd color - did not work so well

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333 --reduceLR_patience 8 --D_init_sd 0.002 \
#     --val_batch_size 16 --beta 5.0  --ad_scale_min 1.5 --ad_scale_max 1.50001 --sigma_chi 0.1 --init_tau 1.0 --ad_bgcolor 127 \
#     --with_rotation True

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333 --reduceLR_patience 8 --D_init_sd 0.002 \
#     --val_batch_size 16 --beta 5.0  --ad_scale_min 1.5 --ad_scale_max 1.50001 --sigma_chi 0.1 --init_tau 1.0 --ad_bgcolor 127 \
#     --with_rotation True
#v33: fixed bgcolor, with rotation


# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333 --D_init_sd 0.002 \
#     --val_batch_size 16 --ad_bgcolor 127 --beta 5.0  --ad_scale_min 1.5 --ad_scale_max 1.50001 --sigma_chi 0.1 --init_tau 1.0 \
#     --new_first_action_inf True --learning_rate 1e-4 --random_seed 1238

# python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD False --batch_size 8 --reduceLR_factor 0.333333 --D_init_sd 0.002 \
    # --val_batch_size 16 --beta 5.0  --new_first_action_inf True 
#v53

python C2PO_trainer.py --gpus 1 2 3 4 5 6 7 8  --threeD True --batch_size 8 --reduceLR_factor 0.333333 --D_init_sd 0.002 \
    --val_batch_size 16 --ad_bgcolor 127 --beta 5.0  --ad_scale_min 1.5 --ad_scale_max 1.50001 --sigma_chi 0.1 --init_tau 1.0 \
    --new_first_action_inf True --learning_rate 3e-4  --with_rotation True --ad_bounding_actions True


