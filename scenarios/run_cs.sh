NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
GPUS=($(seq 0 $((NUM_GPUS-1))))
seeds=($(seq 1 $NUM_GPUS))

setting=mixed_domains
method=$1
architectures=("$2") # (resnet50 vit_b_16 swin_b d2v)
options=("$3")

imagenet_c=("cfgs/imagenet_c/${method}.yaml")
for arch in ${architectures[*]}; do
    if [ "$arch" = "resnet50" ]; then
        lr=2.0e-4
    elif [ "$arch" = "d2v" ]; then
        lr=0.1e-4
    else
        lr=2.5e-4
    fi
    (
    trap 'kill 0' SIGINT; \
    for seed in ${seeds[*]}; do
        for var in "${imagenet_c[@]}"; do
            save_dir="./output/cs/${arch}/${delta}_${method}_seed${seed}"
            rm -rf $save_dir
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python test_time.py --cfg $var \
                SETTING $setting RNG_SEED $seed \
                MODEL.ARCH $arch OPTIM.LR $lr \
                SAVE_DIR $save_dir $options & \
            i=$((i + 1))
        done
    done
    wait
    ) 
done
python summary_results.py --root_path $PWD --setting cs --dataset imagenet_c --method ${method} --models ${architectures[*]} > ./output/cs/cs_${method}_imagenet_c.res


imagenet_others=("cfgs/imagenet_others/${method}.yaml")
dataset=(imagenet_d109)
for ds in ${dataset[*]}; do
    for arch in ${architectures[*]}; do
        if [ "$arch" = "resnet50" ]; then
            lr=2.0e-4
        elif [ "$arch" = "d2v" ]; then
            lr=0.1e-4
        else
            lr=2.5e-4
        fi
        (
        trap 'kill 0' SIGINT; \
        for seed in ${seeds[*]}; do
            for var in "${imagenet_others[@]}"; do
                save_dir="./output/cs/${arch}/${delta}_${method}_${ds}_seed${seed}"
                rm -rf $save_dir
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python test_time.py --cfg $var \
                    SETTING $setting RNG_SEED $seed CORRUPTION.DATASET $ds \
                    MODEL.ARCH $arch OPTIM.LR $lr \
                    SAVE_DIR $save_dir $options & \
                i=$((i + 1))
            done
        done
        wait
        ) 
        
    done
    python summary_results.py --root_path $PWD --setting cs --dataset ${ds} --method ${method} --models ${architectures[*]} --tag ${ds} > ./output/cs/cs_${method}_${ds}.res
done