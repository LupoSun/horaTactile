#!/bin/bash
GPUS=$1
CACHE=$2
C=outputs/AllegroHandHora/"${CACHE}"/stage1_nn/best.pth
CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroHandHora headless=True \
task.env.numEnvs=20000 test=True task.on_evaluation=True \
task.env.object.type=cylinder_default+custom_cylinder_2dcross+custom_cylinder_3dcross \
'task.env.object.sampleProb=[0.34,0.33,0.33]' \
task.env.hora.useShapePrivInfo=True task.env.hora.useExtendedPrivInfo=True \
task.env.hora.privInfoDim=17 train.ppo.use_shape_priv_info=True train.ppo.priv_info_dim=17 \
train.algo=PPO \
task.env.randomization.randomizeMass=True \
task.env.randomization.randomizeCOM=True \
task.env.randomization.randomizeFriction=True \
task.env.randomization.randomizePDGains=True \
task.env.randomization.randomizeScale=True \
task.env.randomization.jointNoiseScale=0.005 \
task.env.reset_height_threshold=0.6 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.ppo.priv_info=True \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint="${C}"
