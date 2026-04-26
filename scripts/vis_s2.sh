#!/bin/bash
CACHE=$1
python train.py task=AllegroHandHora headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object.type=cylinder_default+custom_cylinder_2dcross+custom_cylinder_3dcross \
'task.env.object.sampleProb=[0.34,0.33,0.33]' \
task.env.hora.useShapePrivInfo=True task.env.hora.useExtendedPrivInfo=True \
task.env.hora.privInfoDim=17 train.ppo.use_shape_priv_info=True train.ppo.priv_info_dim=17 \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage2_nn/model_last.ckpt
