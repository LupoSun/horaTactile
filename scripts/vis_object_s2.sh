#!/bin/bash
RUN_NAME=$1
OBJECT_TYPE=$2

if [ -z "${RUN_NAME}" ] || [ -z "${OBJECT_TYPE}" ]; then
  echo "Usage: bash scripts/vis_object_s2.sh <run_name> <object_type>"
  echo "Example: bash scripts/vis_object_s2.sh hora_v0.0.2 custom_btg13_mean"
  exit 1
fi

python train.py task=AllegroHandHora headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object.type="${OBJECT_TYPE}" \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=AllegroHandHora/"${RUN_NAME}" \
checkpoint=outputs/AllegroHandHora/"${RUN_NAME}"/stage2_nn/model_last.ckpt
