"""
统一评测入口: eval.py

推理 (infer.py) 只负责把结果存成 jsonl (每行 {"video_name", "pred", "gt"}),
评测全部在本文件完成。

可选指标 (-m, 可多选):
  acc    -- 准确率 (字符串精确匹配 + 混淆矩阵)
  err    -- 错误率 (100 - acc)
  wer    -- 词错误率
  hit    -- 说话人定位击中率 / 未击中率 (IoU > 0.2)
  loc    -- 说话人定位 Mean IoU
  sawer  -- Identity-Fixed SA-WER, who said what
  ier    -- Identity-Fixed IER, who said when

任务预设 (--task, 与 -m 二选一):
  asr / vsr / avsr  -> wer
  si                -> acc err
  sl                -> hit
  sv                -> acc err
  vrsdr             -> sawer ier

用法:
  python eval.py results/benchmark_si_easy.jsonl --task si
  python eval.py results/benchmark_vr_sdr_full.jsonl --task vrsdr
  python eval.py results/lrs3_asr.jsonl -m wer
"""
import argparse
import glob
import json
import re
import string
from collections import OrderedDict

import numpy as np
from jiwer import process_words, wer as jiwer_wer
from pyannote.core import Annotation, Segment
from pyannote.metrics.identification import IdentificationErrorRate
from sklearn.metrics import confusion_matrix

_TIME_PATTERN = re.compile(r'\[(\d+\.\d+)[-–](\d+\.\d+)s?\]\s*$')


# ---------------------------------------------------------------------------
# 基础解析
# ---------------------------------------------------------------------------

def _clean_text(text):
    text = text.strip()
    text = re.sub(r'\.+\s*$', '.', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def parse_transcript(gt_str):
    """解析 transcript JSON 字符串, 返回 [(speaker, text), ...], 时间戳被去除."""
    try:
        data = json.loads(gt_str)
    except Exception:
        return []
    result = []
    for seg in data.get("transcript", []):
        seg_clean = re.sub(r'\[\d+\.\d+[-–]\d+\.\d+s?\]\s*$', '', seg).strip()
        if ':' not in seg_clean:
            continue
        spk, txt = seg_clean.split(':', 1)
        spk, txt = spk.strip(), re.sub(r'\[[^\]]*\]', '', txt).strip()
        if spk and txt:
            result.append((spk, _clean_text(txt)))
    return result


def parse_segments(gt_str):
    """解析 transcript JSON 字符串, 返回 [(speaker, (start, end)), ...]."""
    try:
        data = json.loads(gt_str)
    except Exception:
        return []
    result = []
    for seg in data.get("transcript", []):
        seg = seg.strip()
        match = _TIME_PATTERN.search(seg)
        if not match:
            continue
        start, end = float(match.group(1)), float(match.group(2))
        content = seg[:match.start()].strip()
        if ':' not in content:
            continue
        spk = content.split(':', 1)[0].strip()
        if spk and start <= end:
            result.append((spk, (start, end)))
    return result


def _merge_by_speaker(utterances):
    """同一说话人的多段文本按出现顺序合并为一条."""
    speaker_texts = OrderedDict()
    for speaker, text in utterances:
        speaker_texts[speaker] = speaker_texts.get(speaker, "") + " " + text
    return {spk: _clean_text(txt) for spk, txt in speaker_texts.items()}


def _normalize(text):
    """去标点、小写, 并把 (a/b) 形式的备选文本取第二个."""
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return re.sub(r'\(([^()/]+)/([^()/]+)\)', r'\2', text)


# ---------------------------------------------------------------------------
# acc: 准确率
# ---------------------------------------------------------------------------

def clc_acc(pred_strs, label_strs):
    acc_num = 0
    for pred, label in zip(pred_strs, label_strs):
        if pred == label:
            acc_num += 1
    acc = 100.0 * acc_num / len(pred_strs)

    labels = sorted(set(label_strs))
    cm = confusion_matrix(label_strs, pred_strs, labels=labels)
    print("Confusion Matrix:")
    print(cm)
    print("Classes: ", labels)
    print("X-axis corresponds to predicted labels")
    print("Y-axis corresponds to true labels")
    print(f"acc: {acc}")
    return acc


def clc_err(pred_strs, label_strs):
    err = 100.0 * sum(p != l for p, l in zip(pred_strs, label_strs)) / len(pred_strs)
    print(f"err: {err}")
    return err


# ---------------------------------------------------------------------------
# wer: 词错误率
# ---------------------------------------------------------------------------

def clc_wer(pred_strs, label_strs):
    value = 100 * jiwer_wer(label_strs, pred_strs)
    print(f"wer: {value}")
    return value


# ---------------------------------------------------------------------------
# loc: 说话人定位 IoU
# ---------------------------------------------------------------------------

def extract_bbox(s):
    s = s.strip()
    if s.startswith('(') and s.endswith(')'):
        s = s[1:-1]
    elif s.startswith('(') and ')' in s:
        s = s.split(')')[0]
        s = s[1:]
    else:
        raise ValueError("字符串必须以括号包围，例如 '(0.1,0.2,0.3,0.4)'")

    coords = [float(x.strip()) for x in s.split(',')]

    if len(coords) != 4:
        raise ValueError(f"需要4个坐标值，但得到 {len(coords)} 个: {s}")

    x1, y1, x2, y2 = coords
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    return (x1, y1, x2, y2)


def calculate_iou_box(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # GT 框标注不够精确, 取较小框面积作分母 (overlap ratio), 保证击中率可靠
    union_area = min(area1, area2)

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calculate_loc(gt_labels_list, pred_labels_list):
    ious = []
    print("Calculating IoU Loc ...")
    for gt_str, pred_str in zip(gt_labels_list, pred_labels_list):
        try:
            gt_box = extract_bbox(gt_str)
            pred_box = extract_bbox(pred_str)
            iou = calculate_iou_box(gt_box, pred_box)
            ious.append(iou)
        except Exception:
            ious.append(0.0)

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    hit_iou2 = [x for x in ious if x > 0.5]
    hit_iou = [x for x in ious if x > 0.2]

    print(f"Overall Mean IoU Loc: {mean_iou:.4f}")
    print(f"总共有{len(ious)}个样本，其中击中样本数量0.2@{len(hit_iou)} 0.5@{len(hit_iou2)}个")

    return mean_iou, ious


def calculate_hit(gt_labels_list, pred_labels_list, iou_threshold=0.2):
    """击中率 / 未击中率: IoU > 阈值记为击中 (默认 0.2)."""
    _, ious = calculate_loc(gt_labels_list, pred_labels_list)

    total = len(ious)
    hit_num = sum(1 for x in ious if x > iou_threshold)
    hit_rate = 100.0 * hit_num / total if total else 0.0
    miss_rate = 100.0 - hit_rate

    print(f"Hit Rate (IoU>{iou_threshold}): {hit_rate:.2f}% ({hit_num}/{total})")
    print(f"Miss Rate:                       {miss_rate:.2f}%")
    return hit_rate, miss_rate


# ---------------------------------------------------------------------------
# sawer: Identity-Fixed SA-WER (who said what)
# ---------------------------------------------------------------------------

def evaluate_sawer(pred_strs, label_strs, verbose=False):
    """SA-WER: 逐样本按说话人聚合后计算多说话人 WER, 再对所有样本取平均."""
    all_wer, failed = [], 0

    for pred_str, gt_str in zip(pred_strs, label_strs):
        gt_dict = _merge_by_speaker(parse_transcript(gt_str))
        pred_dict = _merge_by_speaker(parse_transcript(pred_str))
        if not gt_dict or not pred_dict:
            failed += 1
            continue

        total_sub = total_del = total_ins = total_words = 0
        try:
            for spk in set(gt_dict) | set(pred_dict):
                reference = _normalize(gt_dict.get(spk, ""))
                hypothesis = _normalize(pred_dict.get(spk, ""))
                if verbose:
                    print(f"  Speaker [{spk}]\n    REF: {reference}\n    HYP: {hypothesis}")
                res = process_words(reference, hypothesis)
                total_sub += res.substitutions
                total_del += res.deletions
                total_ins += res.insertions
                total_words += res.hits + res.substitutions + res.deletions
        except Exception as e:
            if verbose:
                print(f"  skipped: {e}")
            failed += 1
            continue

        if total_words == 0:
            sample_wer = 0.0 if total_ins == 0 else float('inf')
        else:
            sample_wer = (total_sub + total_del + total_ins) / total_words
        all_wer.append(sample_wer)

    if not all_wer:
        return None
    sawer = float(np.mean(all_wer))
    print(f"SA-WER (who said what): {sawer * 100:.2f}%  [{len(all_wer)} valid, {failed} skipped]")
    return sawer


# ---------------------------------------------------------------------------
# ier: Identity-Fixed IER (who said when)
# ---------------------------------------------------------------------------

def evaluate_ier(pred_strs, label_strs, verbose=False):
    """IER: 逐样本计算 IdentificationErrorRate(collar=0.25), 取平均."""
    all_ier, failed = [], 0

    for pred_str, gt_str in zip(pred_strs, label_strs):
        gt_segs = parse_segments(gt_str)
        pred_segs = parse_segments(pred_str)
        if not gt_segs or not pred_segs:
            failed += 1
            continue

        ref, hyp = Annotation(), Annotation()
        for name, (start, end) in gt_segs:
            ref[Segment(start, end)] = name
        for name, (start, end) in pred_segs:
            hyp[Segment(start, end)] = name

        ier = IdentificationErrorRate(collar=0.25)(ref, hyp)
        all_ier.append(ier)
        if verbose:
            print(f"  IER: {ier:.2%}")

    if not all_ier:
        return None
    mean_ier = float(np.mean(all_ier))
    print(f"IER (who said when):    {mean_ier * 100:.2f}%  [{len(all_ier)} valid, {failed} skipped]")
    return mean_ier


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

METRICS = {
    "acc": lambda gts, preds, verbose: clc_acc(preds, gts),
    "err": lambda gts, preds, verbose: clc_err(preds, gts),
    "wer": lambda gts, preds, verbose: clc_wer(preds, gts),
    "hit": lambda gts, preds, verbose: calculate_hit(gts, preds),
    "loc": lambda gts, preds, verbose: calculate_loc(gts, preds),
    "sawer": lambda gts, preds, verbose: evaluate_sawer(preds, gts, verbose),
    "ier": lambda gts, preds, verbose: evaluate_ier(preds, gts, verbose),
}

TASK_PRESETS = {
    "asr": ["wer"],
    "vsr": ["wer"],
    "avsr": ["wer"],
    "si": ["acc", "err"],
    "sl": ["hit"],
    "sv": ["acc", "err"],
    "vrsdr": ["sawer", "ier"],
}


def load_results(json_file):
    preds, gts = [], []
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            preds.append(item.get("pred", ""))
            gts.append(item.get("gt", ""))
    return preds, gts


def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for inference result jsonl files")
    parser.add_argument("files", nargs="+", help="Result JSONL files (每行 {video_name, pred, gt})")
    parser.add_argument("-m", "--metrics", nargs="+", choices=list(METRICS), default=None,
                        help="要计算的指标, 可多选")
    parser.add_argument("-t", "--task", choices=list(TASK_PRESETS), default=None,
                        help="任务预设, 自动选择该任务对应的指标")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.metrics is None and args.task is None:
        parser.error("必须指定 -m/--metrics 或 -t/--task 之一")
    metrics = args.metrics if args.metrics is not None else TASK_PRESETS[args.task]

    file_list = []
    for pattern in args.files:
        matched = glob.glob(pattern)
        if matched:
            file_list.extend(sorted(matched))
        else:
            print(f"File not found: {pattern}")

    print(f"{'=' * 78}")
    for file_path in file_list:
        preds, gts = load_results(file_path)
        print(f"File: {file_path}  ({len(preds)} samples)")
        for name in metrics:
            METRICS[name](gts, preds, args.verbose)
        print(f"{'-' * 78}")


if __name__ == "__main__":
    main()
