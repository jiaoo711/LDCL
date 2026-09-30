# coding=utf-8
"""dataset for CUG-UAV
"""
from __future__ import absolute_import, division, unicode_literals

import os
import json
import torch
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import Dataset
from .decode import RawVideoExtractorpyAV
import logging


class CUGUAV_DataLoader(Dataset):
    """CUG-UAV dataset loader."""
    def __init__(
            self,
            csv_path,
            features_path,
            tokenizer,
            max_words=77,
            feature_framerate=1.0,
            max_frames=100,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            lmdb_dataset=None,
            geo_caption_path=None,
            gps_path=None,
            time_caption_path=None,
            time_path=None
    ):
        self.data = pd.read_csv(csv_path)
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.lmdb_dataset = lmdb_dataset

        # --- multi-sentence per video detection ---
        # eval_epoch 使用 multi_sentence_ 分支的条件：dataset.multi_sentence_per_video == True
        # CSV 中同一 video_id 出现多行时自动启用（如每视频 5 条描述）
        # 要求：CSV 中同一视频的多行必须连续排列（相同 video_id 的行紧邻）
        all_video_ids = list(self.data['video_id'].values)
        unique_videos = list(dict.fromkeys(all_video_ids))   # 保序去重
        self.video_num = len(unique_videos)
        self.sentence_num = len(all_video_ids)
        if self.sentence_num > self.video_num:
            self.multi_sentence_per_video = True
            vid_count: dict = {}
            for vid in all_video_ids:
                vid_count[vid] = vid_count.get(vid, 0) + 1
            cumsum = 0
            self.cut_off_points: list = []
            for vid in unique_videos:
                cumsum += vid_count[vid]
                self.cut_off_points.append(cumsum)   # 1-indexed 累计句子数
        else:
            self.multi_sentence_per_video = False
            self.cut_off_points = list(range(1, self.sentence_num + 1))

        self.geo_captions = None
        if geo_caption_path is not None and os.path.exists(geo_caption_path):
            with open(geo_caption_path, 'r', encoding='utf-8') as f:
                self.geo_captions = json.load(f)
        self.gps_data = None
        if gps_path is not None and os.path.exists(gps_path):
            with open(gps_path, 'r', encoding='utf-8') as f:
                self.gps_data = json.load(f)

        self.time_captions = None
        if time_caption_path is not None and os.path.exists(time_caption_path):
            with open(time_caption_path, 'r', encoding='utf-8') as f:
                self.time_captions = json.load(f)
        self.time_data = None
        if time_path is not None and os.path.exists(time_path):
            with open(time_path, 'r', encoding='utf-8') as f:
                self.time_data = json.load(f)

        # 0: ordinary order; 1: reverse order; 2: random order.
        self.frame_order = frame_order
        assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution, is_train=False,
                                                        num_segments=self.max_frames,
                                                        lmdb_dataset=self.lmdb_dataset)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return len(self.data)

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        # pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        # pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        # pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            
            
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words
            
            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        # video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)
        video_list = []
        # video data
        for i, video_id in enumerate(choice_video_ids):
            # torch.Tensor of shape [T, C, H, W]
            # raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(
            #                             os.path.join(self.features_path, "{}.mp4".format(video_id))
            #                             )
            #LZC
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            # 判断路径是否为str类型，如果不是则转换为str类型
            # print(video_path)
            if not isinstance(video_path, str):
                video_path = video_path.decode('utf-8')  # 转换字节路径为字符串
                # print(video_path)
                # input()
            raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(video_path)
            # slice_len = raw_video_data.size(0)
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
            video_list.append(raw_video_data)
        # video mask
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
        # torch.Tensor of shape [Pair, T, C, H, W]
        video = torch.stack(video_list, dim=0)

        return video, video_mask

    def _get_geo_text(self, video_id, sentence_idx=0):
        """将指定视频的地理描述文本 tokenize，无数据时返回全零。"""
        k = 1
        geo_text  = np.zeros((k, self.max_words), dtype=np.int64)
        geo_mask  = np.zeros((k, self.max_words), dtype=np.int64)

        geo_sentence = ""
        if self.geo_captions is not None:
            geo_val = self.geo_captions.get(str(video_id), [])
            if isinstance(geo_val, str):
                geo_sentence = geo_val
            elif geo_val:
                idx = min(sentence_idx, len(geo_val) - 1)
                geo_sentence = geo_val[idx]

        if geo_sentence:
            words = self.tokenizer.tokenize(geo_sentence)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            if len(words) > self.max_words - 1:
                words = words[:self.max_words - 1]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            mask = [1] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                mask.append(0)
            geo_text[0] = np.array(input_ids)
            geo_mask[0] = np.array(mask)
        return geo_text, geo_mask

    def _get_gps(self, video_id):
        """返回视频 GPS 坐标 [lon, lat]，无数据时返回零向量。"""
        if self.gps_data is not None:
            coords = self.gps_data.get(str(video_id))
            if coords is not None:
                return np.array(coords, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _get_time_text(self, video_id, sentence_idx=0):
        """将指定视频的时间描述文本 tokenize，无数据时返回全零。"""
        k = 1
        time_text = np.zeros((k, self.max_words), dtype=np.int64)
        time_mask = np.zeros((k, self.max_words), dtype=np.int64)

        time_sentence = ""
        if self.time_captions is not None:
            time_val = self.time_captions.get(str(video_id), [])
            if isinstance(time_val, str):
                time_sentence = time_val
            elif time_val:
                idx = min(sentence_idx, len(time_val) - 1)
                time_sentence = time_val[idx]

        if time_sentence:
            words = self.tokenizer.tokenize(time_sentence)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            if len(words) > self.max_words - 1:
                words = words[:self.max_words - 1]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            mask = [1] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                mask.append(0)
            time_text[0] = np.array(input_ids)
            time_mask[0] = np.array(mask)
        return time_text, time_mask

    def _get_time(self, video_id):
        """返回视频时间元数据 [month, day, hour, minute, second]，无数据时返回零向量。"""
        if self.time_data is not None:
            t = self.time_data.get(str(video_id))
            if t is not None:
                return np.array(t, dtype=np.float32)
        return np.zeros(5, dtype=np.float32)

    def __getitem__(self, idx):
        video_id = self.data['video_id'].values[idx]
        sentence = self.data['sentence'].values[idx]

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        geo_text, geo_mask = self._get_geo_text(video_id)
        gps_coords = self._get_gps(video_id)
        time_text, time_mask = self._get_time_text(video_id)
        time_coords = self._get_time(video_id)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, geo_text, geo_mask, gps_coords, time_text, time_mask, time_coords


class CUGUAV_TrainDataLoader(Dataset):
    """CUG-UAV train dataset loader."""
    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            tokenizer,
            max_words=77, # 30
            feature_framerate=1.0,
            max_frames=100,
            unfold_sentences=False,
            image_resolution=224,
            frame_order=0,
            slice_framepos=0,
            lmdb_dataset=None,
            geo_caption_path=None,
            gps_path=None,
            time_caption_path=None,
            time_path=None,
            use_dual_gran=False,
            dual_gran_level_threshold=3
    ):
        """
        CUG-UAV training dataset.
        =============================
        Args:
            csv_path: path to the video list
            json_path: text corpus
            features_path: video path
            tokenizer: text tokenizer
            feature_framerate: fps
        """
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.features_path = features_path
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.lmdb_dataset = lmdb_dataset
        self.use_dual_gran = use_dual_gran
        self.dual_gran_level_threshold = dual_gran_level_threshold

        # 0: ordinary order; 1: reverse order; 2: random order.
        # self.frame_order = frame_order
        # assert self.frame_order in [0, 1, 2]
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        self.slice_framepos = slice_framepos
        assert self.slice_framepos in [0, 1, 2]

        self.geo_captions = None
        if geo_caption_path is not None and os.path.exists(geo_caption_path):
            with open(geo_caption_path, 'r', encoding='utf-8') as f:
                self.geo_captions = json.load(f)
        else:
            logging.warning(f"[TrainDataLoader] geo_caption_path not loaded: "
                            f"path={geo_caption_path}, exists={geo_caption_path is not None and os.path.exists(geo_caption_path)}")
        self.gps_data = None
        if gps_path is not None and os.path.exists(gps_path):
            with open(gps_path, 'r', encoding='utf-8') as f:
                self.gps_data = json.load(f)
        else:
            logging.warning(f"[TrainDataLoader] gps_path not loaded: "
                            f"path={gps_path}, exists={gps_path is not None and os.path.exists(gps_path)}")

        self.time_captions = None
        if time_caption_path is not None and os.path.exists(time_caption_path):
            with open(time_caption_path, 'r', encoding='utf-8') as f:
                self.time_captions = json.load(f)
        else:
            logging.warning(f"[TrainDataLoader] time_caption_path not loaded: "
                            f"path={time_caption_path}, exists={time_caption_path is not None and os.path.exists(time_caption_path)}")
        self.time_data = None
        if time_path is not None and os.path.exists(time_path):
            with open(time_path, 'r', encoding='utf-8') as f:
                self.time_data = json.load(f)
        else:
            logging.warning(f"[TrainDataLoader] time_path not loaded: "
                            f"path={time_path}, exists={time_path is not None and os.path.exists(time_path)}")

        logging.info(f"[TrainDataLoader] geo_captions: {len(self.geo_captions) if self.geo_captions else 0} entries, "
                     f"gps_data: {len(self.gps_data) if self.gps_data else 0} entries, "
                     f"time_captions: {len(self.time_captions) if self.time_captions else 0} entries, "
                     f"time_data: {len(self.time_data) if self.time_data else 0} entries")

        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        self.video_l1_captions = {}
        if self.unfold_sentences:
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            video_sentence_counts = defaultdict(int)
            _level_counts = defaultdict(int)
            _geo_text_nonempty = 0
            _gps_nonempty = 0
            _time_text_nonempty = 0
            _time_nonempty = 0
            for itm in self.data['sentences']:
                if itm['video_id'] in train_video_ids:
                    vid = itm['video_id']
                    level = itm.get('level', 1)
                    _level_counts[level] += 1
                    geo_text = ""
                    if self.geo_captions is not None:
                        geo_val = self.geo_captions.get(str(vid), [])
                        if isinstance(geo_val, str):
                            geo_text = geo_val
                        else:
                            sent_idx = video_sentence_counts[vid]
                            if sent_idx < len(geo_val):
                                geo_text = geo_val[sent_idx]
                    if geo_text:
                        _geo_text_nonempty += 1
                    if self.gps_data is not None and self.gps_data.get(str(vid)) is not None:
                        _gps_nonempty += 1
                    time_text = ""
                    if self.time_captions is not None:
                        time_val = self.time_captions.get(str(vid), [])
                        if isinstance(time_val, str):
                            time_text = time_val
                        elif time_val:
                            t_idx = video_sentence_counts[vid]
                            if t_idx < len(time_val):
                                time_text = time_val[t_idx]
                    if time_text:
                        _time_text_nonempty += 1
                    if self.time_data is not None and self.time_data.get(str(vid)) is not None:
                        _time_nonempty += 1
                    self.sentences_dict[len(self.sentences_dict)] = (vid, itm['caption'], geo_text, time_text, level)
                    video_sentence_counts[vid] += 1
                    try:
                        level_int = int(level)
                    except (ValueError, TypeError):
                        level_int = 0
                    if level_int == 1 and vid not in self.video_l1_captions:
                        self.video_l1_captions[vid] = itm['caption']
            self.sample_len = len(self.sentences_dict)
            _dual_eligible = sum(v for k, v in _level_counts.items() if str(k).isdigit() and int(k) >= self.dual_gran_level_threshold)
            logging.info(f"[TrainDataLoader] total_samples={self.sample_len}, "
                         f"level_distribution={dict(_level_counts)}, "
                         f"dual_gran_threshold={self.dual_gran_level_threshold}, "
                         f"dual_eligible_samples={_dual_eligible}, "
                         f"videos_with_L1_caption={len(self.video_l1_captions)}")
            logging.info(f"[TrainDataLoader] geo_text_nonempty={_geo_text_nonempty}/{self.sample_len}, "
                         f"gps_nonempty={_gps_nonempty}/{self.sample_len}, "
                         f"time_text_nonempty={_time_text_nonempty}/{self.sample_len}, "
                         f"time_nonempty={_time_nonempty}/{self.sample_len}")
        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.data['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.data['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

        self.rawVideoExtractor = RawVideoExtractorpyAV(size=image_resolution,
                                                        num_segments=self.max_frames,
                                                        lmdb_dataset=self.lmdb_dataset)
        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    def __len__(self):
        return self.sample_len

    def get_lengths(self):
        """预计算所有样本的有效 token 长度，供 BucketBatchSampler 使用。
        
        结果被缓存为 self._token_lengths，首次调用后后续调用直接返回缓存。
        仅执行文本 tokenize，不加载视频，速度很快。
        """
        if hasattr(self, '_token_lengths'):
            return self._token_lengths

        lengths = []
        for idx in range(self.sample_len):
            if self.unfold_sentences:
                _, caption, _, _, _ = self.sentences_dict[idx]
            else:
                video_id = self.csv['video_id'].values[idx]
                caps = self.sentences.get(video_id, [])
                caption = caps[0] if caps else ''

            words = self.tokenizer.tokenize(caption) if caption else []
            # +2 表示 CLS 和 SEP 特殊 token，上限为 max_words
            length = min(len(words) + 2, self.max_words)
            lengths.append(length)

        self._token_lengths = lengths
        return lengths

    def _get_text(self, video_id, caption=None):
        """get text"""
        k = 1
        choice_video_ids = [video_id]
        # pairs_text = np.zeros((k, self.max_words), dtype=np.long)
        # pairs_mask = np.zeros((k, self.max_words), dtype=np.long)
        # pairs_segment = np.zeros((k, self.max_words), dtype=np.long)
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_words
            assert len(input_mask) == self.max_words
            assert len(segment_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        # video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)
        video_list = []
        # video data
        for i, video_id in enumerate(choice_video_ids):
            # torch.Tensor of shape [T, C, H, W]
            # raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(
            #                         os.path.join(self.features_path, "{}.mp4".format(video_id))
            #                         )
            #LZC
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            # # 判断路径是否为str类型，如果不是则转换为str类型
            # logging.info(video_path)
            # if not isinstance(video_path, str):
            #     try:
            #         video_path = video_path.decode('utf-8')  # 转换字节路径为字符串
            #     except:
            #         pass
            # logging.info(video_path)
            #     # input()
            raw_video_data, slice_len = self.rawVideoExtractor.get_video_data(video_path)
            # slice_len = raw_video_data.size(0)
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
            video_list.append(raw_video_data)
        # vide mask
        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length
        # torch.Tensor of shape [Pair, T, C, H, W]
        video = torch.stack(video_list, dim=0)

        return video, video_mask

    def _geo_text_to_ids(self, geo_sentence):
        """将地理文本字符串 tokenize 为 (geo_text, geo_mask) 数组。"""
        k = 1
        geo_text = np.zeros((k, self.max_words), dtype=np.int64)
        geo_mask = np.zeros((k, self.max_words), dtype=np.int64)
        if geo_sentence:
            words = self.tokenizer.tokenize(geo_sentence)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            if len(words) > self.max_words - 1:
                words = words[:self.max_words - 1]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            mask = [1] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                mask.append(0)
            geo_text[0] = np.array(input_ids)
            geo_mask[0] = np.array(mask)
        return geo_text, geo_mask

    def _get_gps(self, video_id):
        """返回视频 GPS 坐标 [lon, lat]，无数据时返回零向量。"""
        if self.gps_data is not None:
            coords = self.gps_data.get(str(video_id))
            if coords is not None:
                return np.array(coords, dtype=np.float32)
        return np.zeros(2, dtype=np.float32)

    def _time_text_to_ids(self, time_sentence):
        """将时间文本字符串 tokenize 为 (time_text, time_mask) 数组。"""
        k = 1
        time_text = np.zeros((k, self.max_words), dtype=np.int64)
        time_mask = np.zeros((k, self.max_words), dtype=np.int64)
        if time_sentence:
            words = self.tokenizer.tokenize(time_sentence)
            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            if len(words) > self.max_words - 1:
                words = words[:self.max_words - 1]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]
            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            mask = [1] * len(input_ids)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                mask.append(0)
            time_text[0] = np.array(input_ids)
            time_mask[0] = np.array(mask)
        return time_text, time_mask

    def _get_time(self, video_id):
        """返回视频时间元数据 [month, day, hour, minute, second]，无数据时返回零向量。"""
        if self.time_data is not None:
            t = self.time_data.get(str(video_id))
            if t is not None:
                return np.array(t, dtype=np.float32)
        return np.zeros(5, dtype=np.float32)

    def __getitem__(self, idx):
        if self.unfold_sentences:
            video_id, caption, geo_sentence, time_sentence, level = self.sentences_dict[idx]
        else:
            video_id = self.csv['video_id'].values[idx]
            caption = None
            level = 1
            geo_sentence = ""
            time_sentence = ""
            if self.sentences.get(video_id):
                sent_idx = random.randint(0, len(self.sentences[video_id]) - 1)
                caption = self.sentences[video_id][sent_idx]
                if self.geo_captions is not None:
                    geo_val = self.geo_captions.get(str(video_id), [])
                    if isinstance(geo_val, str):
                        geo_sentence = geo_val
                    elif sent_idx < len(geo_val):
                        geo_sentence = geo_val[sent_idx]
                if self.time_captions is not None:
                    time_val = self.time_captions.get(str(video_id), [])
                    if isinstance(time_val, str):
                        time_sentence = time_val
                    elif time_val:
                        t_idx = min(sent_idx, len(time_val) - 1)
                        time_sentence = time_val[t_idx]
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawvideo(choice_video_ids)
        geo_text, geo_mask = self._geo_text_to_ids(geo_sentence)
        gps_coords = self._get_gps(video_id)
        time_text, time_mask = self._time_text_to_ids(time_sentence)
        time_coords = self._get_time(video_id)

        if self.use_dual_gran:
            if int(level) >= self.dual_gran_level_threshold and video_id in self.video_l1_captions:
                short_caption = self.video_l1_captions[video_id]
                short_text, short_mask, short_segment, _ = self._get_text(video_id, short_caption)
            else:
                short_text = np.zeros((1, self.max_words), dtype=np.int64)
                short_mask = np.zeros((1, self.max_words), dtype=np.int64)
                short_segment = np.zeros((1, self.max_words), dtype=np.int64)
        else:
            short_text = np.zeros((1, self.max_words), dtype=np.int64)
            short_mask = np.zeros((1, self.max_words), dtype=np.int64)
            short_segment = np.zeros((1, self.max_words), dtype=np.int64)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask, geo_text, geo_mask, gps_coords, short_text, short_mask, short_segment, time_text, time_mask, time_coords
