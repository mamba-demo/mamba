CUDA=2

exp='MIL_OTLS_single_down--4x'
feats_path='/home/andrew/data/OTLS_single_down--4x/patch_128-32_step_128-8_3D_top5_global/2plus1d_h5_patch_features/imagenet_mean_HE'
result_path='/home/andrew/workspace/results'
clinical_path='/home/andrew/workspace/ThreeDimPlayground/csv/OTLS/prostate_clinical_list_original_single.csv'

lr=0.0002
scheduler='cosine'
aug=0

encoder='2plus1d'
epochs=50
epochs_finetune=0
loss='bce'

lr_finetune=0.0001
split_mode='kf'
split_fold=5
weight_decay=5e-4
attn_dim=64

numOfclasses=2
task='clf'

prop_train=1
dec_enc_dim=256
dec_enc_num=1
grad_accum=10
optim='adamw'
slice_mode='slice'
sample_prop=0.5

dropout=0.5
seed_exp=100
for seed_data in 100 200 300 400 500
do
  for context_network in 'GRU'
  do
    for decoder in 'attn'
    do
      for sample_prop in 0.5
      do
        for grad_accum in 10
        do
              CUDA_VISIBLE_DEVICES=$CUDA python ../train_cv.py --loss $loss --numOfclasses $numOfclasses --context --context_network $context_network --gated --exp $exp \
              --scheduler $scheduler --weight_decay $weight_decay --split_mode $split_mode --split_fold $split_fold \
              --lr $lr --opt $optim --grad_accum $grad_accum --dropout $dropout \
              --sample_mode $slice_mode --decoder_enc --decoder_enc_dim $dec_enc_dim --decoder_enc_num $dec_enc_num \
              --prop_train $prop_train --epochs $epochs --encoder $encoder --decoder $decoder --task $task \
              --numOfaug $aug --seed_data $seed_data --seed_exp $seed_exp --config conf/config_MIL_OTLS.yaml \
              --sample_prop $sample_prop --attn_latent_dim $attn_dim \
              --feats_path $feats_path --result_path $result_path --clinical_path $clinical_path \
              --epochs_finetune $epochs_finetune --lr_finetune $lr_finetune
        done
      done
    done
  done
done

