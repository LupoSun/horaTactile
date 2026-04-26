#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroHandHora headless=True seed=${SEED} \
task.env.numEnvs=20000 \
task.env.object.type=cylinder_default+custom_cylinder_2dcross+custom_cylinder_3dcross \
'task.env.object.sampleProb=[0.34,0.33,0.33]' \
task.env.hora.useShapePrivInfo=True task.env.hora.useExtendedPrivInfo=True \
task.env.hora.privInfoDim=17 train.ppo.use_shape_priv_info=True train.ppo.priv_info_dim=17 \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage1_nn/best.pth \
${EXTRA_ARGS}
