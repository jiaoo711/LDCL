# coding=utf-8
import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_retrieval import CUGUAV_DataLoader
from dataloaders.dataloader_retrieval import CUGUAV_TrainDataLoader

def dataloader_cug_uav_train(args, tokenizer):
    cug_uav_dataset = CUGUAV_TrainDataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        unfold_sentences=args.expand_cug_uav_sentences,
        frame_order=args.train_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset,
        geo_caption_path=getattr(args, 'geo_caption_path', None),
        gps_path=getattr(args, 'gps_path', None),
        time_caption_path=getattr(args, 'time_caption_path', None),
        time_path=getattr(args, 'time_path', None),
        use_dual_gran=getattr(args, 'use_dual_gran', 0),
        dual_gran_level_threshold=getattr(args, 'dual_gran_level_threshold', 3)
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(cug_uav_dataset)
        dataloader = DataLoader(
            cug_uav_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            pin_memory=False,
            sampler=train_sampler,
            drop_last=True,
        )
    else:
        train_sampler = None
        dataloader = DataLoader(
            cug_uav_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            pin_memory=False,
            shuffle=True,
            drop_last=True,
        )

    return dataloader, len(cug_uav_dataset), train_sampler

def dataloader_cug_uav_test(args, tokenizer, subset="test"):
    cug_uav_testset = CUGUAV_DataLoader(
        csv_path=args.val_csv,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        frame_order=args.eval_frame_order,
        slice_framepos=args.slice_framepos,
        lmdb_dataset=args.lmdb_dataset,
        geo_caption_path=getattr(args, 'geo_caption_path', None),
        gps_path=getattr(args, 'gps_path', None),
        time_caption_path=getattr(args, 'time_caption_path', None),
        time_path=getattr(args, 'time_path', None)
    )
    dataloader_cug_uav = DataLoader(
        cug_uav_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_cug_uav, len(cug_uav_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["cug_uav"] = {"train":dataloader_cug_uav_train, "val":dataloader_cug_uav_test, "test":None}
