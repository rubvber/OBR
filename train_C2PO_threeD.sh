#! /bin/sh

python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  \
--resume_from_checkpoint /home/rubber/C2PO/C2PO_logs/lightning_logs/version_3/checkpoints/last.ckpt
#v1, v2, v3, v8

# python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  --n_latent 32 --learning_rate 1e-4
#Didn't work

# python C2PO_trainer.py --gpus 4 5 6 7  --threeD True --batch_size 16 --reduceLR_factor 0.333333  --reduceLR_patience 8 --with_rotation True
#v7
