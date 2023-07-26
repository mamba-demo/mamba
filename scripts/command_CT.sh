CUDA=2

exp='MIL_CT_trimmed_final'

feats_path='/home/andrew/data/microCT_trimmed/patch_128-32_step_128-32_3D_all_global/2plus1d_h5_patch_features/2plus1d_low_25k_top_1pct_mean_HE'
result_path='/home/andrew/workspace/results'

clinical_path='/home/andrew/workspace/ThreeDimPlayground/csv/CT/prostate_clinical_list_train_surv.csv'
lr=0.0002
scheduler='cosine'

aug=0

encoder='resnet50_2d'
#encoder='slowfast'
epochs=50
epochs_finetune=0
lr_finetune=0.0001
split_mode='kf'
split_fold=5
weight_decay=5e-4
context_network='GRU'

numOfclasses=2
task='clf'
loss='bce'

prop_train=1
sample_prop=0.5
dec_enc_num=1
dec_enc_dim=256
grad_accum=10
penalty_gp=0
length_scale=50
optim='adamw'
sample_mode='slice'
dropout=0.5

seed_exp=100
for seed_data in 300 400 500
do
  for decoder in 'attn'
  do
    for attn_dim in 64
    do
      CUDA_VISIBLE_DEVICES=$CUDA python ../train_cv.py --loss $loss --numOfclasses $numOfclasses --context_network $context_network --gated --exp $exp \
      --scheduler $scheduler --weight_decay $weight_decay --split_mode $split_mode --split_fold $split_fold \
      --lr $lr --opt $optim --grad_accum $grad_accum --dropout $dropout \
      --sample_mode $sample_mode  --decoder_enc --decoder_enc_dim $dec_enc_dim --decoder_enc_num $dec_enc_num \
      --prop_train $prop_train --epochs $epochs --encoder $encoder --decoder $decoder --task $task \
      --numOfaug $aug --seed_data $seed_data --seed_exp $seed_exp --config conf/config_MIL_CT.yaml \
      --sample_prop $sample_prop --attn_latent_dim $attn_dim \
      --feats_path $feats_path --result_path $result_path --clinical_path $clinical_path \
      --epochs_finetune $epochs_finetune --lr_finetune $lr_finetune
    done
  done
done