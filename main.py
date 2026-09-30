# coding=utf-8
from __future__ import (absolute_import, division, unicode_literals)

import os
import sys
import time
import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch import optim, distributed
# from torch.cuda.amp import GradScaler
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from params import get_args, save_hp_to_json
from modules import CLIP4Clip, convert_weights
from modules import SimpleTokenizer as ClipTokenizer
from modules.file import PYTORCH_PRETRAINED_BERT_CACHE
from dataloaders.data_dataloaders import DATALOADER_DICT
from utils.lr_scheduler import lr_scheduler
from utils.optimization import BertAdam, prep_optim_params_groups
from utils.log import setup_primary_logging, setup_worker_logging
from utils.misc import set_random_seed, convert_models_to_fp32, save_checkpoint
from utils.dist_utils import is_master, get_rank, is_dist_avail_and_initialized, init_distributed_mode
from utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

best_R1 = 0

def main(args):
    """main function"""

    set_random_seed(args.seed)

    # Set multiprocessing type to spawn.
    if not args.remote:
        torch.multiprocessing.set_start_method('spawn')

    # Set logger
    log_queue = setup_primary_logging(os.path.join(args.output_dir, "log.txt"), args.log_level, args.remote)

    # lmdb
    if args.lmdb_dataset not in [None, 'None']:
        assert os.path.exists(args.lmdb_dataset)
        print('INFO: [dataset] Using {} as data source'.format(args.lmdb_dataset))

    # the number of gpus
    args.ngpus_per_node = torch.cuda.device_count()
    print("INFO: [CUDA] The number of GPUs in this node is {}".format(args.ngpus_per_node))

    # Distributed training = training on more than one GPU.
    # Also easily possible to extend to multiple nodes & multiple GPUs.
    args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
    if args.distributed:
        if args.remote:
            raise NotImplementedError
        else:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, log_queue, args))
    else:
        # nn.DataParallel (DP)
        if args.dp:
            args.gpu, args.world_size = args.multigpu[0], len(args.multigpu)
        else:
            args.world_size = 1
        main_worker(args.gpu, None, log_queue, args)


def main_worker(gpu, ngpus_per_node, log_queue, args):
    """main worker"""
    global best_R1
    args.gpu = gpu


    ## ####################################
    # initilization
    ## ####################################
    global_rank = init_distributed_mode(args, ngpus_per_node, gpu)
    setup_worker_logging(global_rank, log_queue, args.log_level)
    # Lock the random seed of the model to ensure that the model initialization of each process is the same.
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # save parameters
    if is_master(): save_hp_to_json(args.output_dir, args)


    ## ####################################
    # create model
    ## ####################################
    # create tokenizer
    tokenizer = ClipTokenizer()
    model_state_dict = torch.load(args.init_model, map_location='cpu') if args.init_model else None
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    
    model = CLIP4Clip.from_pretrained(args.cross_model, 
                                        cache_dir=cache_dir,
                                        state_dict=model_state_dict,
                                        task_config=args)
   
    #model.freeze_cip_layers(args.freeze_layer_num)
    
    #logging.info('\nweight from DeepCluster')
     

    # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
    if args.precision == "amp"or args.gpu is None:  	# or args.precision == "fp32" 
        logging.info("[weight convert] ==>> Convert weights to fp32 for {}...".format(args.precision))
        convert_models_to_fp32(model)
        logging.info("[weight convert] ==>> Convert done!")

    if not torch.cuda.is_available():
        model.float()
        logging.warning("using CPU, this will be slow")
    else:
        model.cuda(args.gpu)
        if args.precision == "fp16":
            convert_weights(model)
        # Previously batch size and workers were global and not per GPU.
        # args.batch_size = args.batch_size / ngpus_per_node)
        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        if args.distributed and args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.distributed:
            if args.freeze_clip:
               print('do freeze_clip:',args.freeze_clip)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                                find_unused_parameters=True if args.freeze_clip else False)
        if args.dp:
            model = torch.nn.DataParallel(model, device_ids=args.multigpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", get_rank() % ngpus_per_node)


    ## ####################################
    # dataloader loading
    ## ####################################
    assert args.datatype in DATALOADER_DICT
    assert DATALOADER_DICT[args.datatype]["test"] is not None \
           or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
    num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                    / args.gradient_accumulation_steps) * args.epochs


    ## ####################################
    # optimization strategies
    ## ####################################
    optimizer_grouped_parameters = prep_optim_params_groups(args, model, coef_lr=args.coef_lr)
    scaler = GradScaler() if args.precision == "amp" else None
    if args.optim == 'BertAdam':
        logging.info('[optimizer] Using BertAdam Optimizer...')
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                                schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                                t_total=num_train_optimization_steps, weight_decay=args.wd,
                                max_grad_norm=1.0)
        scheduler = None
    elif args.optim == 'AdamW':
        logging.info('[optimizer] Using AdamW Optimizer...')
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.lr,
                                betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.wd)
        scheduler = lr_scheduler(mode='cos', init_lr=args.lr, all_iters=num_train_optimization_steps,
                                    slow_start_iters=args.warmup_proportion * num_train_optimization_steps,
                                    weight_decay=args.wd
                                )
    else:
        raise NotImplementedError

    if is_master():
        tf_writer = SummaryWriter(args.tensorboard_path)
    else:
        tf_writer = None

    ## ####################################
    #  optionally resume from a checkpoint
    ## ####################################
    start_epoch, global_step = 0, 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)

            if not args.load_from_pretrained:
                if "optimizer" in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "scaler" in checkpoint and scaler is not None:
                    logging.info("[resume] => Loading state_dict of AMP loss scaler")
                    scaler.load_state_dict(checkpoint['scaler'])
                start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]

            logging.info(f"[resume] => loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})\n")
        else:
            logging.info("[resume] => no checkpoint found at '{}'\n".format(args.resume))


    ## ####################################
    # train and evalution
    ## ####################################
    if is_master():
        logging.info("\n======================== Running training ========================")
        logging.info("  Num examples = %d", train_length)
        logging.info("  Batch size = %d", args.batch_size)
        logging.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
        logging.info("\n======================== Running test ========================")
        logging.info("  Num examples = %d", test_length)
        logging.info("  Batch size = %d", args.batch_size_val)
        logging.info("  Num steps = %d", len(test_dataloader))
        logging.info("\n======================== Running val ========================")
        logging.info("  Num examples = %d", val_length)

    all_start = time.time()

    if args.do_eval and is_master():
        R1, infer_epoch_time, info_str = eval_epoch(model, test_dataloader, device, args=args)
        torch.cuda.synchronize()
        all_time = time.time() - all_start
        logging.info('The total running time of the program is {:.2f} Seconds\n'.format(all_time))
        logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
                        torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
        sys.exit(0)

    eval_infer_times = []
    best_e = 0
    best_info = []
    for epoch in range(start_epoch, args.epochs):
        if is_dist_avail_and_initialized():
            train_sampler.set_epoch(epoch)
        # set_random_seed(epoch + args.seed)

        tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, optimizer,
                                            global_step, scaler=scaler, tf_writer=tf_writer, scheduler=scheduler)

        if is_master():
            logging.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)

            # Run on val dataset, this process is *TIME-consuming*.
            R1, infer_epoch_time, info_str = eval_epoch(model, test_dataloader, device, args=args, epoch=epoch)
            eval_infer_times.append(infer_epoch_time)
            if best_R1 <= R1:
                best_R1 = R1
                best_e = epoch
                best_info = info_str
            logging.info("The best R1 is: {:.4f}, best_e={}\n".format(best_R1, best_e))
            # save checkpoint
            ckpt_dict = {
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'arch': 'CLIp4Clip',
                    'state_dict': model.state_dict(),
                    'best_acc1': best_R1,
                    'optimizer': optimizer.state_dict(),
                }
            if scaler is not None: ckpt_dict['scaler'] = scaler.state_dict()
            save_checkpoint(ckpt_dict, best_R1 <= R1, args.output_dir, filename='ckpt.pth.tar')

        # synchronise all ranks: non-master ranks must wait here while master
        # runs eval_epoch + checkpoint save, otherwise they race ahead into the
        # next epoch's all_gather and trigger an NCCL timeout.
        if is_dist_avail_and_initialized():
            torch.distributed.barrier()

    all_time = time.time() - all_start

    if is_master():
        logging.info('The total running time of the program is {:.1f} Hour {:.1f} Minute\n'.format(all_time // 3600, 
                    all_time % 3600 / 60))
        logging.info('The average inference time of {} runs is {:.2f} Seconds\n'.format(args.epochs, np.mean(eval_infer_times)))
        logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
                    torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
        logging.info("The best R1 is: {:.4f}, best_epoch={}\n".format(best_R1, best_e))
        for info in best_info:
            logging.info(info)
        print("The above program id is {}\n".format(args.output_dir))

    torch.cuda.empty_cache()
    sys.exit(0)


def train_epoch(epoch, args, model, train_dataloader, device, optimizer, global_step,
                scheduler=None, scaler=None, tf_writer=None):
    samples_per_epoch = len(train_dataloader.dataset)

    # torch.cuda.empty_cache()
    model.train()

    if epoch == 0 and is_master():
        no_clip = args.new_added_modules
        trainable_size =0
        total_param_size  = 0  
        for name, param in model.module.named_parameters():
            if param.requires_grad==True:
                total_param_size += param.numel() 
                trainable_size += param.numel() 
                param_size_MB = param.numel()/(1000**2)
                # logging.info(f'trainerble parameters are: {name}, size is {param_size_MB:.4f} MB')
            else:
                total_param_size += param.numel() 
        trainable_size_MB = trainable_size/(1000**2)
        total_param_size_MB = total_param_size/(1000**2)
        percentage = (trainable_size / total_param_size)*100
        # logging.info("Trainable param percentage are: {}".format(percentage))
        # logging.info("Trainable params are: {} MB, Total params are: {} MB".format(trainable_size_MB,total_param_size_MB))


    total_loss = 0

    end = time.time()

    for step, batch in enumerate(train_dataloader):
        # if step == 1:
        #     break
        # logging.info(f"Step: {step}   Batch: {batch}")
        
        optimizer.zero_grad()
        if scheduler is not None: scheduler(optimizer, global_step=global_step)
        # multi-gpu does scattering it-self
        if torch.cuda.is_available():
            batch = tuple(t.to(device=device, non_blocking=True) for t in batch)
        input_ids, input_mask, segment_ids, video, video_mask, geo_text, geo_mask, gps_coords, short_ids, short_mask, short_segment, time_text, time_mask, time_coords = batch
        data_time = time.time() - end
        
        # logging.info(f"Step: {step}   input_ids: {input_ids}")
        # logging.info(f"Step: {step}   input_mask: {input_mask}")
        # logging.info(f"Step: {step}   segment_ids: {segment_ids}")
        # logging.info(f"Step: {step}   video: {video}")
        # logging.info(f"Step: {step}   video_mask: {video_mask}")

        # forward
        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(input_ids, segment_ids, input_mask, video, video_mask,
                           geo_text, geo_mask, gps_coords,
                           short_ids, short_mask, short_segment,
                           time_text, time_mask, time_coords)
            loss = output['loss'].mean()
            sim_loss = output['sim_loss'].mean()
            dual_loss = output['dual_gran_loss'].mean() if output.get('dual_gran_loss') is not None else None
            geo_loss = output['geo_loss'].mean() if output.get('geo_loss') is not None else None
            time_loss = output['time_loss'].mean() if output.get('time_loss') is not None else None

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
            
        # logging.info(f"Step: {step}   Loss: {loss}")
        # logging.info(f"Step: {step}   Sim_Loss: {sim_loss}")

        # update weights
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.clip_grad_norm is not None:
                    # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, 0.1, 4.6052)
        else:
            torch.clamp_(model.clip.logit_scale.data, 0.1, 4.6052)

        batch_time = time.time() - end
        end = time.time()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            global_step += 1
            if global_step % args.n_display == 0 and is_master():
                num_samples = (step + 1) * len(input_ids) * args.world_size
                percent_complete = num_samples * 1.0 / samples_per_epoch * 100
                logit_scale_data = model.module.clip.logit_scale.data if hasattr(model, 'module') \
                                    else model.clip.logit_scale.data
                # param_groups[0] 是 CLIP backbone（lr_mult=coef_lr），显示会偏低
                # param_groups[2] 是新增参数组（lr_mult=1.0），才是实际训练 LR
                if args.optim == 'AdamW':
                    _new_group_idx = 2 if len(optimizer.param_groups) > 2 else 0
                    lr_tmp = optimizer.param_groups[_new_group_idx]['lr']
                else:
                    lr_tmp = optimizer.get_lr()[0]
                
                _aux_str = ""
                if dual_loss is not None:
                    _aux_str += f"DualLoss: {dual_loss.item():.4f} "
                if geo_loss is not None:
                    _aux_str += f"GeoLoss: {geo_loss.item():.4f} "
                if time_loss is not None:
                    _aux_str += f"TimeLoss: {time_loss.item():.4f} "
                logging.info(
                    f"Epoch: {epoch} [{num_samples} ({percent_complete:.1f}%)]\t"
                    f"SimLoss: {sim_loss.item():.4f} {_aux_str}\t"
                    f"Data (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}"
                    f"\tLR: {lr_tmp:.1e}\tlogit_scale {logit_scale_data:.3f}"
                )
                # tensorboard log
                log_data = {
                    "sim_loss": sim_loss.item(),
                    "data_time": data_time,
                    "batch_time": batch_time,
                    "scale": logit_scale_data.item(),
                    "lr": lr_tmp
                }
                if dual_loss is not None:
                    log_data["dual_loss"] = dual_loss.item()
                if geo_loss is not None:
                    log_data["geo_loss"] = geo_loss.item()
                if time_loss is not None:
                    log_data["time_loss"] = time_loss.item()
                for name, val in log_data.items():
                    name = "train/" + name
                    if tf_writer is not None:
                        tf_writer.add_scalar(name, val, global_step=global_step)

        total_loss += float(loss)

    total_loss = total_loss / len(train_dataloader)

    return total_loss, global_step


def eval_epoch(model, test_dataloader, device, args=None, epoch=0):
    """evaluation"""

    # #################################################################
    ## below variables are used to multi-sentences retrieval
    # multi_sentence_: important tag for eval
    # cut_off_points: used to tag the label when calculate the metric
    # sentence_num: used to cut the sentence representation
    # video_num: used to cut the video representation
    # #################################################################
    multi_sentence_ = False
    cut_off_points_, sentence_num_, video_num_ = [], -1, -1
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') \
            and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence_ = True
        cut_off_points_ = test_dataloader.dataset.cut_off_points
        sentence_num_ = test_dataloader.dataset.sentence_num
        video_num_ = test_dataloader.dataset.video_num
        cut_off_points_ = [itm - 1 for itm in cut_off_points_]

    if multi_sentence_ and is_master():
        logging.info("Eval under the multi-sentence per video clip setting.")
        logging.info("sentence num: {}, video num: {}".format(sentence_num_, video_num_))

    model.eval()
    # unwrap DDP wrapper to avoid collective buffer broadcast during eval
    # (only rank 0 runs eval; DDP.forward would trigger broadcast_buffers which
    #  is a collective op and deadlocks because other ranks are at the barrier)
    _model = model.module if hasattr(model, 'module') else model
    # only fuse geo similarity at eval when geo_captions were actually loaded;
    # otherwise all geo_text embeddings are identical (padding) and add a
    # query-independent bias that corrupts the retrieval ranking.
    _has_geo_data = (getattr(args, 'use_geo_branch', 0)
                     and hasattr(test_dataloader.dataset, 'geo_captions')
                     and test_dataloader.dataset.geo_captions is not None)
    _has_time_data = (getattr(args, 'use_time_branch', 0)
                      and hasattr(test_dataloader.dataset, 'time_captions')
                      and test_dataloader.dataset.time_captions is not None)
    with torch.no_grad():
        batch_list_t = []
        batch_list_v = []
        batch_sequence_output_list, batch_visual_output_list = [], []
        batch_geo_text_list, batch_gps_list = [], []
        batch_time_text_list, batch_time_list = [], []
        total_video_num = 0

        # ----------------------------
        # 1. cache the features
        # ----------------------------
        infer_start_t = time.time()
        for bid, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, video, video_mask, geo_text, geo_mask, gps_coords, time_text, time_mask, time_coords = batch
            if args.save_feature_path is not None and os.path.exists(args.save_feature_path):
                if bid < 2000:
                    if args.datatype == 'cug_uav':
                        print('{}\t'.format(bid + 1), test_dataloader.dataset.data['video_id'].values[bid], end='\n')
                    if args.datatype == 'lsmdc':
                        if 'Harry_Potter' in test_dataloader.dataset.iter2video_pairs_dict[bid][0]:
                            print('{}\t'.format(bid + 1), test_dataloader.dataset.iter2video_pairs_dict[bid], end='\n')

            if multi_sentence_:
                # multi-sentences retrieval means: one clip has two or more descriptions.
                b, *_t = video.shape
                # pass geo data alongside text so the geo branch can extract
                # geo_text_feat and gps_feat in the same forward call
                _text_out = _model(input_ids, segment_ids, input_mask,
                                   geo_text=geo_text, geo_mask=geo_mask,
                                   gps_coords=gps_coords,
                                   time_text=time_text, time_mask=time_mask,
                                   time_coords=time_coords)
                sequence_output = _text_out['sequence_output']
                batch_sequence_output_list.append(sequence_output)
                batch_list_t.append((input_mask, segment_ids,))

                # collect geo_text features for every text sample
                if _has_geo_data and _text_out.get('geo_text_output') is not None:
                    batch_geo_text_list.append(_text_out['geo_text_output'])
                # collect time_text features for every text sample
                if _has_time_data and _text_out.get('time_text_output') is not None:
                    batch_time_text_list.append(_text_out['time_text_output'])

                s_, e_ = total_video_num, total_video_num + b
                filter_inds = [itm - s_ for itm in cut_off_points_ if itm >= s_ and itm < e_]

                if len(filter_inds) > 0:
                    video, video_mask = video[filter_inds, ...], video_mask[filter_inds, ...]
                    visual_output = _model(video=video, video_mask=video_mask)['visual_output']
                    batch_visual_output_list.append(visual_output)
                    batch_list_v.append((video_mask,))
                    # GPS features only for unique videos (matching visual batch structure)
                    if _has_geo_data and _text_out.get('gps_output') is not None:
                        batch_gps_list.append(_text_out['gps_output'][filter_inds])
                    # Time features only for unique videos
                    if _has_time_data and _text_out.get('time_output') is not None:
                        batch_time_list.append(_text_out['time_output'][filter_inds])
                total_video_num += b
            else:
                output = _model(input_ids, segment_ids, input_mask, video, video_mask,
                                geo_text, geo_mask, gps_coords,
                                time_text=time_text, time_mask=time_mask,
                                time_coords=time_coords)
                batch_sequence_output_list.append(output['sequence_output'])
                batch_list_t.append((input_mask, segment_ids,))

                batch_visual_output_list.append(output['visual_output'])
                batch_list_v.append((video_mask,))

                if _has_geo_data and output.get('gps_output') is not None:
                    batch_geo_text_list.append(output['geo_text_output'])
                    batch_gps_list.append(output['gps_output'])

                if _has_time_data and output.get('time_output') is not None:
                    batch_time_text_list.append(output['time_text_output'])
                    batch_time_list.append(output['time_output'])

            if (bid + 1) % args.n_display == 0 or ( bid + 1) == len(test_dataloader):
                logging.info("{}/{}\r".format(bid, len(test_dataloader)))

        if torch.cuda.is_available(): torch.cuda.synchronize()
        all_infer_time = time.time() - infer_start_t
        logging.info('The total model inference time of the program is {:.2f} Seconds\n'.format(all_infer_time))
        if args.inference_speed_test:
            return 0

        # ----------------------------------
        # 2. calculate the similarity
        # ----------------------------------
        sim_matrix = _run_on_single_gpu(model, batch_list_t, batch_list_v,
                                            batch_sequence_output_list, batch_visual_output_list,
                                            batch_geo_text_list=batch_geo_text_list if batch_geo_text_list else None,
                                            batch_gps_list=batch_gps_list if batch_gps_list else None,
                                            batch_time_text_list=batch_time_text_list if batch_time_text_list else None,
                                            batch_time_list=batch_time_list if batch_time_list else None,
                                            args=args)

    if multi_sentence_:
        logging.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len_ = [itm + 1 for itm in cut_off_points_]
        max_length = max([e_-s_ for s_, e_ in zip([0]+cut_off_points2len_[:-1], cut_off_points2len_)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
            sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                  np.full((max_length-e_+s_, sim_matrix.shape[1]), -np.inf)), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logging.info("after reshape, sim matrix size: {} x {} x {}".
                    format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))

    else:
        logging.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
        logging.info('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))


    # return for final logging
    info_str = []
    info_str.append("Text-to-Video:")
    t2v_print_str = []
    for k in range(1, 11):
        metric_key = f'R{k}'
        if metric_key in tv_metrics:
            t2v_print_str.append(f"R@{k}: {tv_metrics[metric_key]:.1f}")
    if 'MR' in tv_metrics:
        t2v_print_str.append(f"Median R: {tv_metrics['MR']:.1f}")
    if 'MeanR' in tv_metrics:
        t2v_print_str.append(f"Mean R: {tv_metrics['MeanR']:.1f}")
    info_str.append(' (metric) >>>  ' + ' - '.join(t2v_print_str))

    info_str.append("Video-to-Text:")
    v2t_print_str = []
    for k in range(1, 11):
        metric_key = f'R{k}'
        if metric_key in vt_metrics:
            v2t_print_str.append(f"V2T$R@{k}: {vt_metrics[metric_key]:.1f}")
    if 'MR' in vt_metrics:
        v2t_print_str.append(f"V2T$Median R: {vt_metrics['MR']:.1f}")
    if 'MeanR' in vt_metrics:
        v2t_print_str.append(f"V2T$Mean R: {vt_metrics['MeanR']:.1f}")
    info_str.append(' (metric) >>>  ' + ' - '.join(v2t_print_str))

    for info in info_str: logging.info(info)
    R1 = tv_metrics['R1']

    return R1, all_infer_time, info_str


def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list,
                       batch_geo_text_list=None, batch_gps_list=None,
                       batch_time_text_list=None, batch_time_list=None, args=None):
    """"calculate the similarity between visual output and text output"""
    if hasattr(model, 'module'):
        model = model.module
    else:
        model = model

    use_geo = (batch_geo_text_list is not None and len(batch_geo_text_list) > 0
               and getattr(args, 'use_geo_branch', 0))
    geo_sim_weight = getattr(args, 'geo_sim_weight', 0.3)

    use_time = (batch_time_text_list is not None and len(batch_time_text_list) > 0
                and getattr(args, 'use_time_branch', 0))
    time_sim_weight = getattr(args, 'time_sim_weight', 0.2)

    # 三模态融合时，确保权重总和 ≤ 1
    vt_weight = 1.0 - (geo_sim_weight if use_geo else 0.0) - (time_sim_weight if use_time else 0.0)
    vt_weight = max(vt_weight, 0.0)

    sim_matrix = []
    
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        geo_text_out_i = batch_geo_text_list[idx1] if use_geo and idx1 < len(batch_geo_text_list) else None
        time_text_out_i = batch_time_text_list[idx1] if use_time and idx1 < len(batch_time_text_list) else None
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask)
            # 多模态融合
            if use_geo or use_time:
                fused_logits = vt_weight * b1b2_logits
                if use_geo and geo_text_out_i is not None and idx2 < len(batch_gps_list):
                    gps_out_j = batch_gps_list[idx2]
                    geo_logits = model.get_geo_similarity_logits(geo_text_out_i, gps_out_j)
                    fused_logits = fused_logits + geo_sim_weight * geo_logits
                if use_time and time_text_out_i is not None and idx2 < len(batch_time_list):
                    time_out_j = batch_time_list[idx2]
                    time_logits = model.get_time_similarity_logits(time_text_out_i, time_out_j)
                    fused_logits = fused_logits + time_sim_weight * time_logits
                b1b2_logits = fused_logits
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    
    sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)

    # if args.camoe_dsl:
    # 	print('Apply DSL')
    # 	# https://github.com/starmemda/CAMoE
    # 	sim_matrix_ = torch.from_numpy(sim_matrix)
    # 	sim_matrix = sim_matrix_ * F.softmax(sim_matrix_, dim=0) * len(sim_matrix_)
    # 	# sim_matrix = sim_matrix_ * F.softmax(sim_matrix_, dim=1) * len(sim_matrix_)
    # 	sim_matrix = sim_matrix.cpu().numpy()

    return sim_matrix


if __name__ == "__main__":
    args = get_args()

    main(args)
    