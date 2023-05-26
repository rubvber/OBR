# C2PO & active dSprites

Code for the C2PO architecture and active dSPrites environment. 

## Dependencies:
- python 3.10.4
- matplotlib 3.5.2
- numpy 1.22.3
- pillow 9.2.0
- pycairo 1.21.0
- pytorch 1.12.1 (with cudatoolkit 11.3, torchvision 0.13.1 and torchaudio 0.12.1)
- pytorch lightning 1.7.3
- scikit-learn 1.1.1
- tensorboard 2.10.0

See also the requirements.txt file which may be used to set up a conda environment.

## C2PO
C2PO (or Continuous Control and Planning with Objects) is a recurrent DNN architecture for object-centric perception, action and control that is trained without supervision on raw RGB video frames. It consists of a world model component and a preference network, both of which interface with an object-structured latent representation of the environment. The world model performs perceptual inference, prediction and goal-directed planning. Perception consists of iterative amortized (variational) inference on a Gaussian Mixture model for individual video frames, linked together via a linearized dynamics model with Gaussian noise. These dynamics can be influenced by control actions that accelerate objects. The linearized dynamics model, combined with a latent representation in second-order generalized coordinates, allows for highly efficient prediction and planning, as both can be performed in closed form via linear projections. Planning requires that a goal or preferred state be supplied in the network's latent space, and this is the purpose of the preference network. It takes as input the current beliefs about object states, and outputs a predicted distribution over their preferred states, thus allowing the model to learn tasks defined by abstract rules.


### Training
C2PO is implemented as a Pytorch Lightning module, which amongst other things allows for easy multi-GPU training. To train C2PO's world model (with default settings) on multiple GPUs, you can use (e.g.) one of the following commands:

`python C2PO_trainer.py --gpus 3 4 5 6`

`python C2PO_trainer.py --gpus -4`


The first option would train on 4 gpus with indices 3, 4, 5 and 6 within your setup. The second option (note the negative sign) lets the code pick 4 GPUs automatically. For single-GPU training, use one of the following:

`python C2PO_trainer.py --gpus 0`

`python C2PO_trainer.py --gpus -1`

