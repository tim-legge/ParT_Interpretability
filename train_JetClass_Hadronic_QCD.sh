#!/bin/bash

set -x

source env.sh

echo "args: $@"

# set the dataset dir via `DATADIR_JetClass`
DATADIR=${DATADIR_JetClass}
[[ -z $DATADIR ]] && DATADIR='./datasets/JetClass'

# set a comment via `COMMENT`
suffix=${COMMENT}

# set the number of gpus for DDP training via `DDP_NGPUS`
NGPUS=${DDP_NGPUS}
[[ -z $NGPUS ]] && NGPUS=1
if ((NGPUS > 1)); then
    CMD="torchrun --standalone --nnodes=1 --nproc_per_node=$NGPUS $(which weaver) --backend nccl"
else
    CMD="weaver"
fi

epochs=50
samples_per_epoch=$((10000 * 1024 / $NGPUS))
samples_per_epoch_val=$((10000 * 128))
dataopts="--num-workers 2 --fetch-step 0.01"

# PN, PFN, PCNN, ParT
model=$1
if [[ "$model" == "ParT" ]]; then
    modelopts="networks/example_ParticleTransformer.py --use-amp"
    batchopts="--batch-size 512 --start-lr 1e-3"
elif [[ "$model" == "PN" ]]; then
    modelopts="networks/example_ParticleNet.py"
    batchopts="--batch-size 512 --start-lr 1e-2"
elif [[ "$model" == "PFN" ]]; then
    modelopts="networks/example_PFN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
elif [[ "$model" == "PCNN" ]]; then
    modelopts="networks/example_PCNN.py"
    batchopts="--batch-size 4096 --start-lr 2e-2"
else
    echo "Invalid model $model!"
    exit 1
fi

# "kin", "kinpid", "full", "kin_Hadronic_QCD"
FEATURE_TYPE=$2
[[ -z ${FEATURE_TYPE} ]] && FEATURE_TYPE="full"

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid|kin_Hadronic_QCD)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# currently only Pythia
SAMPLE_TYPE=Pythia

$CMD \
    --data-train "TTBar:/part-vol-3/weaver-core/particle_transformer/datasets/JetClass/Pythia/train_100M/TTBar_00[0-9].root" \
    "ZJetsToNuNu:/part-vol-3/weaver-core/particle_transformer/datasets/JetClass/Pythia/train_100M/ZJetsToNuNu_00[0-9].root" \
    --data-val "TTBar:/part-vol-3/weaver-core/particle_transformer/datasets/JetClass/Pythia/val_5M/TTBar_120.root" \
    "ZJetsToNuNu:/part-vol-3/weaver-core/particle_transformer/datasets/JetClass/Pythia/val_5M/ZJetsToNuNu_120.root" \
    --data-test \
    "/part-vol-3/weaver-core/particle_transformer/datasets/JetClass/Pythia/test_20M/TTBar_[100-101].root" \
    "/part-vol-3/weaver-core/particle_transformer/datasets/JetClass/Pythia/test_20M/ZJetsToNuNu_[100-101].root" \
    --data-config data/JetClass/JetClass_${FEATURE_TYPE}.yaml --network-config $modelopts \
    --model-prefix training/JetClass/${SAMPLE_TYPE}/${FEATURE_TYPE}/${model}/{auto}${suffix}/net \
    $dataopts $batchopts \
    --samples-per-epoch ${samples_per_epoch} --samples-per-epoch-val ${samples_per_epoch_val} --num-epochs $epochs --gpus 0 \
    --optimizer ranger --log logs/JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}_{auto}${suffix}.log --predict-output pred.root \
    --tensorboard JetClass_${SAMPLE_TYPE}_${FEATURE_TYPE}_${model}${suffix} \
    "${@:3}"
