# coding=utf-8
"""
retrieve_vis.py  -  Interactive text-to-video retrieval + frame visualization.

Usage (same args as training, plus a few extra):
    python retrieve_vis.py \
        --do_train 0 --do_eval 0 \
        --resume logs/<dir>/ckpt.best.pth.tar \
        --val_csv /path/to/test.csv \
        --data_path /path/to/data.json \
        --features_path /path/to/videos/ \
        --pretrained_dir /path/to/pretrained \
        --output_dir /tmp/vis_tmp \
        [all other training args kept identical] \
        --top_k 5 \
        --vis_frames 6 \
        --vis_output_dir vis_output

After video features are cached, type any text query to retrieve Top-K videos.
Results are saved as PNG in --vis_output_dir. Type 'quit' to exit.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# ── matplotlib (non-interactive backend for server use) ──────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import av
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Strip our extra args BEFORE calling get_args() so argparse doesn't complain
# ─────────────────────────────────────────────────────────────────────────────
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--top_k',          type=int, default=5,            help='Number of top results to show')
_pre.add_argument('--vis_frames',     type=int, default=6,            help='Frames sampled per video for display')
_pre.add_argument('--vis_output_dir', type=str, default='vis_output', help='Directory to save result figures')
_vis_args, _remaining = _pre.parse_known_args()
# params.py requires at least one of do_train/do_eval to be True.
# Force do_eval=1 so the validation passes; retrieve_vis never uses the eval loop.
_has_eval  = any(a.startswith('--do_eval')  for a in _remaining)
_has_train = any(a.startswith('--do_train') for a in _remaining)
if not _has_eval and not _has_train:
    _remaining += ['--do_eval', '1']
elif _has_eval:
    # replace --do_eval 0 with --do_eval 1
    for _j, _a in enumerate(_remaining):
        if _a == '--do_eval' and _j + 1 < len(_remaining):
            _remaining[_j + 1] = '1'
            break
        if _a.startswith('--do_eval='):
            _remaining[_j] = '--do_eval=1'
            break
sys.argv = [sys.argv[0]] + _remaining

from params import get_args
from modules import CLIP4Clip
from modules import SimpleTokenizer as ClipTokenizer
from modules.file import PYTORCH_PRETRAINED_BERT_CACHE
from dataloaders.data_dataloaders import DATALOADER_DICT
from utils.misc import convert_models_to_fp32


# ─────────────────────────────────────────────────────────────────────────────
# Text tokenization  (mirrors CUGUAV_DataLoader._get_text)
# ─────────────────────────────────────────────────────────────────────────────
_CLS = "<|startoftext|>"
_SEP = "<|" + "endoftext|>"   # constructed to avoid XML-parser conflicts


def _tokenize(query: str, tokenizer, max_words: int):
    """Tokenize free-text query → (input_ids, input_mask, segment_ids) tensors."""
    words = tokenizer.tokenize(query)
    words = [_CLS] + words
    if len(words) > max_words - 1:
        words = words[:max_words - 1]
    words = words + [_SEP]

    input_ids   = tokenizer.convert_tokens_to_ids(words)
    input_mask  = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    while len(input_ids) < max_words:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return (torch.tensor([input_ids],   dtype=torch.long),
            torch.tensor([input_mask],  dtype=torch.long),
            torch.tensor([segment_ids], dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# Video frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_frames(video_path: str, num_frames: int = 6, size: int = 224):
    """Uniformly sample frames from a video; returns list of PIL.Image."""
    blank = [Image.new('RGB', (size, size), (100, 100, 100))] * num_frames
    if not os.path.exists(video_path):
        return blank
    try:
        container = av.open(video_path)
        stream    = container.streams.video[0]
        if stream.frames:
            total = stream.frames
        else:
            total = sum(1 for _ in container.decode(video=0))
            container.close()
            container = av.open(video_path)      # reopen after counting

        step = max(total // num_frames, 1)
        targets = set(step * i for i in range(num_frames))
        frames, idx = [], 0
        for frame in container.decode(video=0):
            if idx in targets:
                img = frame.to_image().resize((size, size), Image.BILINEAR)
                frames.append(img)
                if len(frames) == num_frames:
                    break
            idx += 1
        container.close()
        while len(frames) < num_frames:
            frames.append(Image.new('RGB', (size, size), (100, 100, 100)))
        return frames
    except Exception:
        return blank


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def _save_figure(query: str, results, features_path: str,
                 num_frames: int, out_dir: str) -> str:
    """
    results: list of (rank:int, video_id:str, score:float)
    Saves a grid image and returns the output path.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_rows = len(results)
    n_cols = num_frames + 1        # col-0 = label, col 1..N = frames

    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.4 + 0.8))
    gs  = gridspec.GridSpec(n_rows + 1, n_cols, hspace=0.35, wspace=0.05)

    # ── title row ────────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, f'Query:  "{query}"',
                  ha='center', va='center', fontsize=11,
                  fontweight='bold', wrap=True,
                  bbox=dict(boxstyle='round,pad=0.4', fc='#f0f4ff', ec='#8888cc'))

    # ── result rows ──────────────────────────────────────────────────────────
    for row, (rank, vid_id, score) in enumerate(results, start=1):
        # label cell
        ax_lbl = fig.add_subplot(gs[row, 0])
        ax_lbl.axis('off')
        color = '#e8f5e9' if rank == 1 else '#ffffff'
        ax_lbl.set_facecolor(color)
        ax_lbl.text(0.5, 0.5,
                    f'#{rank}\n{vid_id}\n{score:.3f}',
                    ha='center', va='center', fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=0.3', fc=color, ec='#aaaaaa'))

        # frame cells
        vpath  = os.path.join(features_path, f"{vid_id}.mp4")
        frames = _extract_frames(vpath, num_frames=num_frames)
        for col, frame in enumerate(frames, start=1):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(np.array(frame))
            ax.axis('off')
            if rank == 1:
                for spine in ax.spines.values():
                    spine.set_edgecolor('#4caf50')
                    spine.set_linewidth(2)
                    spine.set_visible(True)

    safe_q   = "".join(c if (c.isalnum() or c in ' _-') else '_'
                       for c in query)[:50].strip()
    out_path = os.path.join(out_dir, f"{safe_q}.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = ClipTokenizer()

    # ── Build model ──────────────────────────────────────────────────────────
    cache_dir = (args.cache_dir if args.cache_dir
                 else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed'))
    model_state_dict = torch.load(args.init_model, map_location='cpu') if args.init_model else None
    model = CLIP4Clip.from_pretrained(args.cross_model,
                                      cache_dir=cache_dir,
                                      state_dict=model_state_dict,
                                      task_config=args)
    convert_models_to_fp32(model)
    model = model.to(device)

    # ── Load checkpoint ──────────────────────────────────────────────────────
    ckpt_path = getattr(args, 'resume', None)
    if not ckpt_path or str(ckpt_path) == 'None':
        print("[ERROR] Please pass --resume <path/to/ckpt.best.pth.tar>")
        sys.exit(1)
    ckpt   = torch.load(ckpt_path, map_location='cpu')
    sd     = ckpt.get('state_dict', ckpt)
    epoch  = ckpt.get('epoch', '?')
    # Strip DDP 'module.' prefix if model was saved from DistributedDataParallel
    if any(k.startswith('module.') for k in sd):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    print(f"[checkpoint] epoch={epoch}  path={ckpt_path}")
    model.eval()

    # ── Test dataloader ───────────────────────────────────────────────────────
    assert args.datatype in DATALOADER_DICT, f"Unknown datatype: {args.datatype}"
    test_dl, n_test = DATALOADER_DICT[args.datatype]["val"](args, tokenizer)
    ds = test_dl.dataset
    print(f"[data] {n_test} test samples  ({len(ds.data['video_id'].unique())} unique videos)")

    # ── Pre-extract video features ────────────────────────────────────────────
    print("[cache] Extracting video features …")
    feat_list = []
    vid_list  = []
    offset    = 0
    with torch.no_grad():
        for bid, batch in enumerate(test_dl):
            batch = tuple(t.to(device) for t in batch)
            _, _, _, video, video_mask, _, _, _ = batch
            bsz = video.shape[0]

            vout = model(video=video,
                         video_mask=video_mask)['visual_output']   # [B,1,D]
            vout = F.normalize(vout.squeeze(1), dim=-1)            # [B,D]
            feat_list.append(vout.cpu())

            for i in range(bsz):
                idx = offset + i
                if idx < len(ds.data):
                    vid_list.append(str(ds.data['video_id'].values[idx]))
            offset += bsz

            if (bid + 1) % 5 == 0 or (bid + 1) == len(test_dl):
                print(f"  {bid+1}/{len(test_dl)}", end='\r')

    all_feats = torch.cat(feat_list, dim=0)   # [N, D]

    # Deduplicate: keep first occurrence per video_id
    seen, uniq_ids, uniq_feats = set(), [], []
    for vid_id, feat in zip(vid_list, all_feats):
        if vid_id not in seen:
            seen.add(vid_id)
            uniq_ids.append(vid_id)
            uniq_feats.append(feat)
    bank = torch.stack(uniq_feats, dim=0)    # [N_uniq, D]
    print(f"\n[cache] Done.  {len(uniq_ids)} unique videos.")

    # ── Interactive loop ──────────────────────────────────────────────────────
    top_k   = _vis_args.top_k
    n_fr    = _vis_args.vis_frames
    out_dir = _vis_args.vis_output_dir

    print(f"\n[ready] Top-{top_k} retrieval  |  frames={n_fr}  |  out='{out_dir}/'")
    print("        Enter text query (or 'quit' to exit)")
    print("-" * 60)

    while True:
        try:
            query = input("\nQuery > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ('quit', 'exit', 'q'):
            break

        input_ids, input_mask, seg_ids = _tokenize(query, tokenizer, args.max_words)
        input_ids  = input_ids.to(device)
        input_mask = input_mask.to(device)
        seg_ids    = seg_ids.to(device)

        with torch.no_grad():
            tout = model(input_ids, seg_ids, input_mask)['sequence_output']
            tout = F.normalize(tout.squeeze(0).squeeze(0), dim=-1)   # [D]

        scores, idx = (bank.to(device) @ tout).cpu().topk(top_k)

        print(f"\n  Top-{top_k}:")
        results = []
        for rank, (i, s) in enumerate(zip(idx.tolist(), scores.tolist()), 1):
            vid = uniq_ids[i]
            print(f"  #{rank}  {vid}  ({s:.4f})")
            results.append((rank, vid, s))

        out_path = _save_figure(query, results,
                                features_path=args.features_path,
                                num_frames=n_fr,
                                out_dir=out_dir)
        print(f"  [saved] {out_path}")

    print("[done]")


if __name__ == '__main__':
    main()
