NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
GPUS=($(seq 0 $((NUM_GPUS-1))))
seeds=($(seq 1 $NUM_GPUS))

setting=mixed_domains_correlated
method=$1
architectures=("$2") # (vit_b_16 swin_b d2v)
options=("$3")

imagenet_c=("cfgs/imagenet_c/${method}.yaml")
deltas=(0.01)
for arch in ${architectures[*]}; do
    for delta in ${deltas[*]}; do
    if [ "$arch" = "d2v" ]; then
        lr=0.1e-4
        options=()
    else
        lr=2.5e-4
        options=("CMF.Q 0.00025")
    fi
    (
    trap 'kill 0' SIGINT; \
    for seed in ${seeds[*]}; do
        for var in "${imagenet_c[@]}"; do
            save_dir="./output/tc_ls_cs/${arch}/${delta}_${method}_seed${seed}"
            rm -rf $save_dir
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python test_time.py --cfg $var \
                SETTING $setting RNG_SEED $seed TEST.ALPHA_DIRICHLET $delta \
                MODEL.ARCH $arch OPTIM.LR $lr \
                SAVE_DIR $save_dir $options & \
            i=$((i + 1))
        done
    done
    wait
    )
    done
done
python summary_results.py --root_path $PWD --setting tc_ls_cs --dataset imagenet_c --method ${method} --models ${architectures[*]} --deltas ${deltas[*]} > ./output/tc_ls_cs/tc_ls_cs_${method}_imagenet_c.res

imagenet_others=("cfgs/imagenet_others/${method}.yaml")
dataset=(imagenet_d109)
deltas=(0.1)
for ds in ${dataset[*]}; do
    for arch in ${architectures[*]}; do
        for delta in ${deltas[*]}; do
        if [ "$arch" = "d2v" ]; then
            lr=0.1e-4
            options=()
        else
            lr=2.5e-4
            options=("CMF.Q 0.00025")
        fi
        (
        trap 'kill 0' SIGINT; \
        for seed in ${seeds[*]}; do
            for var in "${imagenet_others[@]}"; do
                save_dir="./output/tc_ls_cs/${arch}/${delta}_${method}_${ds}_seed${seed}"
                rm -rf $save_dir
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python test_time.py --cfg $var \
                    SETTING $setting RNG_SEED $seed TEST.ALPHA_DIRICHLET $delta \
                    MODEL.ARCH $arch CORRUPTION.DATASET $ds OPTIM.LR $lr \
                    SAVE_DIR $save_dir $options & \
                i=$((i + 1))
            done
        done
        wait
        )
        done
    done
    python summary_results.py --root_path $PWD --setting tc_ls_cs --dataset ${ds} --method ${method} --models ${architectures[*]} --tag ${ds} --deltas ${deltas[*]} > ./output/tc_ls_cs/tc_ls_cs_${method}_${ds}.res
done
