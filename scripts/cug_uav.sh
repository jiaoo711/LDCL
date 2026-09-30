#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_TIMEOUT=7200000
echo "Using local machine for training"

# 原始参数
group=group2-2

# dataset
dataset=cug_uav
fps=3

DATA_PATH={}

data_path=${DATA_PATH}/Annotations/cug_uav_data.json
train_csv=${DATA_PATH}/Annotations/cug_uav_train.csv
val_csv=${DATA_PATH}/Annotations/cug_uav_test.csv
features_path=${DATA_PATH}/Compressed_videos/
geo_caption_path=${DATA_PATH}/Annotations/video_captions_EN.jsonl   # 地理描述文本JSON路径，格式 {video_id: caption}
gps_path=${DATA_PATH}/Annotations/video_gps.json           # GPS坐标JSON路径，格式 {video_id: [lon, lat]}

pretrained_dir={}

# train or eval
do_train=1
do_eval=0

# learning strategies
pretrained_clip_name=ViT-B/32
lr=1e-3
coef_lr=5e-4
wd=0.2
epochs=5
optim=AdamW
# 消融开关：1=启用上下文扩展(248 token，位置编码插值)，0=保留原始编码器(77 token)
extend_context=1
max_words=248   # extend_context=1时使用248；若设extend_context=0则自动回退到77
# 文本适配器层数消融（需 extend_context=1）：0=无适配器，1=单线性，2=瓶颈MLP，3=深瓶颈MLP
text_adapter_layers=0

# 双粒度联合训练损失开关：1=启用 Long-CLIP 风格粗粒度辅助损失，0=禁用（对照组）
use_dual_gran=1
dual_gran_lambda=0.5       # 粗粒度辅助损失权重（赞数：0.5/1.0/1.5）
dual_gran_smoothing=0.1    # label smoothing（内置值：0.0/0.1）
dual_gran_level_threshold=4  # 触发阈值：level>=3 的样本才计算辅助损失

# GPS地理模态消融开关：1=启用Geo-CLIP分支，0=禁用（对照组）
use_geo_branch=1
geo_pretrained_path={}  # Geo-CLIP LocationEncoder预训练权重路径（.pth），默认weights/location_encoder_weights.pth
geo_loss_weight=0.1     # 地理对比损失融合权重 λ（0.3配合队列梯度干扰严重，0.1更稳定）
geo_sim_weight=0.3      # 推理时地理相似度融合权重 w
gps_queue_size=128     # GPS特征队列大小；对1000-1500样本/batch=16推荐64-256，填满仅需8步
# GPS队列负样本消融开关：1=启用历史GPS队列扩大负样本，0=退回纯in-batch对比损失（对照组）
use_gps_queue=0

max_frames=12

resume=None
load_from_pretrained=0

batch_size=32           # single GPU batch size
batch_size_val=16
num_workers=8

n_display=20            # log per n_display
precision=amp

freeze_clip=1
time_embedding=0

shared_latent_space=linear

# distributed training
init_method='tcp://127.0.0.1:6010'


current_datetime=$(TZ="Asia/Shanghai" date +"%Y-%m-%d-%H:%M:%S")
model_dir=logs/${current_datetime}_${dataset}_STOP
echo "The model dir is ${model_dir}"
# CUDA_LAUNCH_BLOCKING=1
python  main.py \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --num_thread_reader ${num_workers} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --n_display ${n_display} \
        --train_csv ${train_csv} \
        --val_csv ${val_csv} \
        --data_path ${data_path} \
        --features_path ${features_path} \
        --output_dir ${model_dir} \
        --optim ${optim} \
        --lr ${lr} \
        --coef_lr ${coef_lr} \
        --wd ${wd} \
        --max_words ${max_words} \
        --max_frames ${max_frames} \
        --batch_size_val ${batch_size_val} \
        --datatype ${dataset} \
        --expand_cug_uav_sentences  \
        --feature_framerate ${fps} \
        --freeze_layer_num 0  \
        --slice_framepos 2 \
        --loose_type \
        --linear_patch 2d \
        --sim_header meanP \
        --pretrained_clip_name ${pretrained_clip_name} \
        --precision ${precision} \
        --init_method ${init_method} \
        --pretrained_dir ${pretrained_dir} \
        --freeze_clip ${freeze_clip} \
        --time_embedding ${time_embedding} \
        --resume ${resume} \
        --load_from_pretrained ${load_from_pretrained} \
        --shared_latent_space ${shared_latent_space} \
        --temporal_prompt ${group} \
        --extend_context ${extend_context} \
        --text_adapter_layers ${text_adapter_layers} \
        --use_dual_gran ${use_dual_gran} \
        --dual_gran_lambda ${dual_gran_lambda} \
        --dual_gran_smoothing ${dual_gran_smoothing} \
        --dual_gran_level_threshold ${dual_gran_level_threshold} \
        --use_geo_branch ${use_geo_branch} \
        --geo_loss_weight ${geo_loss_weight} \
        --geo_sim_weight ${geo_sim_weight} \
        --geo_caption_path ${geo_caption_path} \
        --gps_path ${gps_path} \
        --geo_pretrained_path ${geo_pretrained_path} \
        --gps_queue_size ${gps_queue_size} \
        --use_gps_queue ${use_gps_queue}

echo "Training Finished!!!"
