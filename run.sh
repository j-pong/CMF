method=$1

bash scenarios/run_cs.sh $1 "resnet50 vit_b_16 swin_b d2v" && \
bash scenarios/run_tc_cs.sh $1 "resnet50 vit_b_16 swin_b d2v" && \
bash scenarios/run_tc_ls_tc_cs.sh $1 "vit_b_16 swin_b d2v" && \
bash scenarios/run_tc_ls_cs.sh $1 "vit_b_16 swin_b d2v"