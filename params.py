#coding=utf-8

import os
import json
import logging
import argparse


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B/32", "ViT-B/16"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def get_args(description='DGL on Retrieval Task'):
    """config of program"""
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--do_pretrain", action='store_true', default=False,
                            help="Whether to run training.")
    parser.add_argument("--do_train", type=int, default=1,
                            help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0,
                            help="Whether to run eval on the dev set.")

    parser.add_argument("--inference_speed_test", type=int, default=0,
                            help="Only test the inference speed.")

    parser.add_argument("--debug", default=False, action="store_true",
                            help="If true, more information is logged.")
    # datasets
    parser.add_argument('--data_dir', type=str, default='/cache/dataset',
                            help='where all data located')

    parser.add_argument('--lmdb_dataset', type=str, default=None,
                            help="LMDB database for the dataset")

    parser.add_argument('--save_feature_path', type=str,
                            default=None,
                            help='Used to save the CLIP features')

    parser.add_argument('--train_csv', type=str, default='data/.train.csv', help='')
    
    parser.add_argument('--val_csv', type=str, default='data/.val.csv', help='')
    
    parser.add_argument('--data_path', type=str, default='data/caption.pickle', help='data pickle file path')

    parser.add_argument('--features_path', type=str, default='data/videos_feature.pickle', help='feature path')

    # training settings
    parser.add_argument('--num_thread_reader', type=int, default=1, help='')

    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')

    parser.add_argument('--batch_size', type=int, default=256, help='batch size')

    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    # learning strategies
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')

    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')

    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')

    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")

    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")

    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")

    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")

    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=248, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                            help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--resume", default=None, type=str,
                            help="path to latest checkpoint (default: none)",)

    parser.add_argument('--load_from_pretrained', type=int, default=0,
                            help="load optimizer and scaler state from pretrained model.")

    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")

    parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")

    parser.add_argument("--optim", default='BertAdam', type=str, choices=['BertAdam', 'AdamW'],
                            help="The optimizer")

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument("--clip_grad_norm", default=1.0, type=float,
                            help="the maximum gradient norm (default None)")

    parser.add_argument("--cache_dir", default="", type=str,
                            help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")

    parser.add_argument("--datatype", default="cug_uav", type=str, help="Point the dataset to finetune.")

    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', action='store_true', 
                            help="Default using tight type for retrieval.")
    parser.add_argument('--expand_cug_uav_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                            help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                            help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                            help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                            help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                            choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                            help="choice a similarity header.")
    
    # setting about pretrained weights
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str,
                            help="Choose a CLIP version")

    parser.add_argument("--pretrained_dir", default=os.path.expanduser("~/models/pretrained"), type=str,
                            help="The pretrained directory of CLIP pretrained model")

    # arguments for distributed training
    parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")

    parser.add_argument('--world_size', default=1, type=int,
                            help='number of nodes for distributed training')

    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")

    parser.add_argument("--init_method", default="tcp://127.0.0.1:6101", type=str,
                               help="url used to set up distributed training")

    # setting about GPUs
    parser.add_argument("--dp", default=False, action="store_true",
                            help="Use DP instead of DDP.")

    parser.add_argument("--multigpu", default=None, type=lambda x: [int(a) for a in x.split(",")],
                            help="In DP, which GPUs to use for multigpu training", )

    parser.add_argument("--gpu", type=int, default=None,
                            help="Specify a single GPU to run the code on for debugging."
                             "Leave at None to use all available GPUs.", )

    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--use-bn-sync", default=False, action="store_true",
                            help="Whether to use batch norm sync.")

    # setting about remote server cluster
    parser.add_argument("--remote", type=int, default=0, help="use remote server cluster or not.")
    # precision of training weights
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="fp32",
                            help="Floating point precition.")


    parser.add_argument('--freeze_clip', type=int, default=0,
                            help="Whether freeze all clip backbone.")

    # divide the pretrained temperature
    parser.add_argument('--temperature_new', type=float, default=1.0,
                            help='assign a new temperature to CLIP model')

    parser.add_argument('--time_embedding', type=int, default=0,
                            help="Add time embedding in CLIP model.")
    # test of DSL loss in CAMOE
    parser.add_argument('--camoe_dsl', type=int, default=0,
                            help="Add DSL loss for CAMOE.")

    parser.add_argument('--pre_norm', type=int, default=0,
                            help="whether do l2 normalization before clustering.")

    ## add for the efficient prompt
    parser.add_argument('--tfm_heads', type=int, default=8)
    parser.add_argument('--tfm_layers', type=int, default=2)


    parser.add_argument('--shared_latent_space', type=str, default='transformer', choices=['transformer', 'linear'])
    parser.add_argument('--lora', type=bool, default=True)
    
    ## add for unified prompt transformer and global visual prompts
    parser.add_argument('--text_prompt_length', type=int, default=8)
    parser.add_argument('--local_each_frame_prompt_length', type=int, default=4)
    parser.add_argument('--global_visual_prompt_length', type=int, default=4)
    parser.add_argument('--unified_transformer_layers', type=int, default=12)

    parser.add_argument('--visual_output_type',  type=str, default='global_prompt0', choices=['global_prompt0', 'average_global_prompt','average_frame_cls_token','average_global_prompt_and_frame_cls_token','global-local-feature'],
                            help="The choice of visual output for abalation")
    
    parser.add_argument('--temporal_prompt', type=str, default="DGL")

    parser.add_argument('--extend_context', type=int, default=1, choices=[0, 1],
                            help="扩展CLIP文本编码器上下文长度：77 → 248 tokens（分段线性插值）。"
                                 "0: 禁用（保留原始77-token编码器，消融对照组），"
                                 "1: 启用（248-token编码器，默认）。")

    parser.add_argument('--text_adapter_layers', type=int, default=2, choices=[0, 1, 2, 3, 4],
                            help="扩展文本编码器后的适配器MLP层数（需 extend_context=1）。"
                                 "0: 无适配器（消融对照），"
                                 "1: 单线性层 D→D，"
                                 "2: 瓶颈MLP D→D/2→D，"
                                 "3: 深瓶颈MLP D→D/2→D/2→D（默认2）。")


    parser.add_argument('--use_bucket_sampler', type=int, default=1, choices=[0, 1],
                            help="按长度分桶采样消融开关。"
                                 "0: 禁用（退回 RandomSampler，对照组），"
                                 "1: 启用 BucketBatchSampler + 动态 Padding（默认，节省显存）。")

    parser.add_argument('--bucket_width', type=int, default=50,
                            help="分桶宽度（token 数），仅在 use_bucket_sampler=1 时生效。"
                                 "较小值桶内长度更均匀（如 30），较大值减少桶数量（如 100）。默认 50。")

    parser.add_argument('--gps_queue_size', type=int, default=65536,
                            help="GPS特征队列大小，用于扩大地理对比损失的负样本覆盖度（参考GeoCLIP队列机制）。"
                                 "仅在 use_geo_branch=1 且 use_gps_queue=1 时生效。建议设为数据集大小的整数倍。")

    parser.add_argument('--use_gps_queue', type=int, default=1, choices=[0, 1],
                            help="GPS队列负样本消融开关（需 use_geo_branch=1 同时开启）。"
                                 "0: 纯in-batch对称损失（对照组，验证队列机制的岂变），"
                                 "1: 启用历史GPS特征队列扩大负样本覆盖度。")

    # GPS geo-modality branch
    parser.add_argument('--use_geo_branch', type=int, default=0, choices=[0, 1],
                            help="启用GPS地理模态分支（Geo-CLIP LocationEncoder）。"
                                 "0: 禁用（用于消融对照），1: 启用。")

    parser.add_argument('--geo_caption_path', type=str, default=None,
                            help="地理描述文本JSON文件路径，格式 {video_id: [caption, ...]}。")

    parser.add_argument('--gps_path', type=str, default=None,
                            help="GPS坐标JSON文件路径，格式 {video_id: [lon, lat]}。")

    parser.add_argument('--geo_pretrained_path', type=str, default=None,
                            help="Geo-CLIP LocationEncoder预训练权重路径（.pth文件）。")

    parser.add_argument('--geo_loss_weight', type=float, default=0.3,
                            help="地理对比损失的融合权重 λ，总损失 = L_vt + λ * L_geo。")

    parser.add_argument('--geo_sim_weight', type=float, default=0.3,
                            help="推理时地理相似度融合权重，最终相似度 = (1-w)*S_vt + w*S_geo。")

    # Time modality branch (GT-Loc TimeEncoder)
    parser.add_argument('--use_time_branch', type=int, default=0, choices=[0, 1],
                            help="启用时间模态分支（GT-Loc TimeEncoder）。"
                                 "0: 禁用（用于消融对照），1: 启用。")

    parser.add_argument('--time_caption_path', type=str, default=None,
                            help="时间描述文本JSON文件路径，格式 {video_id: [caption, ...]}。")

    parser.add_argument('--time_path', type=str, default=None,
                            help="时间元数据JSON文件路径，格式 {video_id: [month, day, hour, minute, second]}。")

    parser.add_argument('--time_pretrained_path', type=str, default=None,
                            help="GT-Loc checkpoint路径（可选），用于提取预训练时间编码器权重。")

    parser.add_argument('--time_loss_weight', type=float, default=0.1,
                            help="时间对比损失的融合权重 λ_time，总损失 = L_vt + λ_geo*L_geo + λ_time*L_time。")

    parser.add_argument('--time_sim_weight', type=float, default=0.2,
                            help="推理时时间相似度融合权重 w_time，最终相似度 = (1-w_geo-w_time)*S_vt + w_geo*S_geo + w_time*S_time。")

    parser.add_argument('--use_time_queue', type=int, default=0, choices=[0, 1],
                            help="时间特征队列负样本消融开关（需 use_time_branch=1）。"
                                 "0: 纯in-batch对称损失，1: 启用历史时间特征队列。")

    parser.add_argument('--time_queue_size', type=int, default=256,
                            help="时间特征队列大小，仅在 use_time_branch=1 且 use_time_queue=1 时生效。")

    parser.add_argument('--time_encoder_sigma', type=str, default='1,16,256',
                            help="时间编码器高斯核标准差列表，逗号分隔。"
                                 "默认 '1,16,256' (三尺度)，可选 '1' (单尺度), '256' (单尺度), '1,256' (双尺度)。")

    parser.add_argument('--time_encoder_simple_mlp', type=int, default=0, choices=[0, 1],
                            help="时间编码器简化模式：0=使用角度表示+RFF（默认），1=仅MLP（消融对照）。")

    # 双粒度联合训练损失（Long-CLIP 风格）
    parser.add_argument('--use_dual_gran', type=int, default=0, choices=[0, 1],
                            help="双粒度联合训练损失开关。"
                                 "1: 对 level >= dual_gran_level_threshold 的样本添加 L1 粗粒度辅助损失。")

    parser.add_argument('--dual_gran_lambda', type=float, default=1.0,
                            help="粗粒度辅助损失的融合权重 λ，总损失 = L_current + λ·L_L1。默认 1.0（等权）。")

    parser.add_argument('--dual_gran_smoothing', type=float, default=0.0,
                            help="粗粒度辅助损失的 label smoothing 系数（参考 Long-CLIP=0.1）。0 表示不使用。")

    parser.add_argument('--dual_gran_level_threshold', type=int, default=3,
                            help="触发粗粒度辅助损失的最低粒度级别（包含），level >= threshold 的样本才计算辅助损失。")

    args = parser.parse_args()

    assert args.task_type == "retrieval"

    if args.sim_header == "tightTransf":
        args.loose_type = False
    if args.datatype == 'activity':
        # pre-pooling to avoid OOM, only work for meanP with AcitivityNet when eval
        args.pre_visual_pooling = 1

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    # Shell 脚本传入 "None" 字符串时规范化为 Python None，避免触发误导性 checkpoint 警告
    if args.resume in ('None', 'none', ''):
        args.resume = None

    args.tensorboard_path = os.path.join(args.output_dir, "tensorboard")
    # logging level
    args.log_level = logging.DEBUG if args.debug else logging.INFO

    # 上下文长度扩展消融：禁用时将max_words回退到原始CLIP上限77
    if not args.extend_context:
        args.max_words = min(args.max_words, 77)

    # new added params
    if args.shared_latent_space == 'transformer':
        args.new_added_modules = ['PromptTransformer','unified_prompt_mlp','unified_prompt_embedding','prompt_proj','prompt_embeddings']
    if args.shared_latent_space == 'linear':
        args.new_added_modules = ['visual_prompt_embedding','prompt_embeddings','prefix_text_prompt_proj_layer','postfix_text_prompt_proj_layer']
    
    if getattr(args, 'extend_context', 0):
        args.new_added_modules += ['text_long_projection']
        if getattr(args, 'text_adapter_layers', 0) > 0:
            args.new_added_modules += ['text_long_adapter', 'text_adapter_dropout']
    args.new_added_modules += ["visual.TemporalPrompt"]
    if args.lora:
        args.new_added_modules += ["LoRA"]
    if getattr(args, 'use_geo_branch', 0):
        args.new_added_modules += ["geo_encoder"]
    if getattr(args, 'use_time_branch', 0):
        args.new_added_modules += ["time_encoder"]

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.pretrained_clip_name)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    print('\n', vars(args), '\n')
    # save_hp_to_json(args.output_dir, args)

    return args


def save_hp_to_json(directory, args):
    """Save hyperparameters to a json file
    """
    filename = os.path.join(directory, 'hparams_train.json')
    hparams = vars(args)
    with open(filename, 'w') as f:
        json.dump(hparams, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    args = get_args()
