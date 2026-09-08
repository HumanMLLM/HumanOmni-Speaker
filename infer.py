"""
推理脚本: infer.py

调用链: infer.py -> delta_encoder/ (delta_encoder.py + resnet.py + qwen_omni_forward.py) + omni_dataset.py
推理只负责生成并保存结果 (每行 {video_name, pred, gt} 的 jsonl),
评测请用: python eval.py <result.jsonl> -t <task>

权重来源 (由 convert_ckpt.py 从旧 merge.ckpt 转换得到):
  HumanOmniSpeaker/delta_encoder.pt  -- DeltaEncoder 权重, 直接加载
  HumanOmniSpeaker/llm_weights.pt    -- Qwen2.5-Omni Thinker 主干权重 (键带 "llm." 前缀)

用法:
  python infer.py [--dataset xxx.jsonl] [--out results/xxx.jsonl] [--max-samples N]
"""
import argparse
import json
import os
import queue
import re
import threading

import torch

from transformers import (
    AutoConfig,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)
from delta_encoder import DeltaEncoder, apply_forward_patch
from omni_dataset import ScriptArgs, OmniSpeakerDataset, CustomCollater

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 显式打猴子补丁: 替换 Thinker.forward / get_rope_index
apply_forward_patch()


# =====================================================================
# 模型加载
# =====================================================================

def load_model(cfg_file, llm_weights_path, delta_weights_path, model_name, use_delta_token=True, device="cuda:0"):
    print(f"Loading base model structure from {model_name}")
    base_config = AutoConfig.from_pretrained(model_name)
    thinker_config = base_config.thinker_config if hasattr(base_config, "thinker_config") else base_config
    model = Qwen2_5OmniThinkerForConditionalGeneration(thinker_config).to(torch.float16).to(device)

    if use_delta_token:
        delta_encoder = DeltaEncoder(cfg_file, weights_path=delta_weights_path)
        delta_encoder = delta_encoder.to(device=model.device, dtype=model.dtype)
        model.delta_encoder = delta_encoder
    else:
        model.delta_encoder = None

    print(f"Loading LLM weights from {llm_weights_path}")
    new_state_dict = torch.load(llm_weights_path, map_location="cpu", weights_only=True)

    current_state_dict = model.state_dict()
    for key in current_state_dict.keys():
        if "llm." + key in new_state_dict:
            current_state_dict[key] = new_state_dict["llm." + key]
        elif not key.startswith("delta_encoder."):
            # delta_encoder 权重已单独加载, 不在此打日志
            print(f"not Updating {key}")

    model.load_state_dict(current_state_dict)
    return model


# =====================================================================
# 工具函数
# =====================================================================

def get_video_path(messages):
    for msg in messages:
        if msg["role"] == "user":
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "video":
                    return item.get("video")
                if isinstance(item, dict) and item.get("type") == "audio":
                    return item.get("audio")
    return None


def extract_answer(text):
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    return match.group(1).strip().lower() if match else ""


# =====================================================================
# 主流程
# =====================================================================

def main(args):
    cfg_file = os.path.join(BASE_DIR, "HumanOmniSpeaker", "delta_encoder.json")
    model_name = os.path.join(BASE_DIR, "HumanOmniSpeaker")

    dataset_name_val = args.dataset

    use_delta_token = args.use_delta_token
    use_vb_token = args.use_vb_token
    use_audio_token = args.use_audio_token
    print(f"token config: use_delta_token={use_delta_token}, use_vb_token={use_vb_token}, use_audio_token={use_audio_token}")

    fps = 25
    min_pixels = 224 * 224
    max_pixels = 235200
    max_new_tokens = 3000
    generate_timeout = args.timeout

    dataset_val = OmniSpeakerDataset(
        dataset_name_val,
        ScriptArgs({
            "video_fps": fps,
            "use_delta_token": use_delta_token,
            "use_vb_token": use_vb_token,
            "use_audio_token": use_audio_token,
        }),
    )

    model = load_model(cfg_file, args.llm_weights, args.delta_weights, model_name, use_delta_token=use_delta_token)
    model.eval()

    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    new_collate_fn = CustomCollater(min_pixels=min_pixels, max_pixels=max_pixels, fps=fps, mode="test")

    indx = 0
    dataset_len = len(dataset_val)
    print(dataset_len)

    max_samples = args.max_samples if args.max_samples > 0 else dataset_len

    # generate 常驻 worker 线程: 单样本超时后等上一条 generate 真正结束,
    # 避免两个 generate 同时占用模型造成竞争
    task_queue = queue.Queue()
    result_queue = queue.Queue()

    def generate_worker():
        while True:
            gen_inputs = task_queue.get()
            try:
                ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                result_queue.put(ids)
            except Exception as e:
                result_queue.put(e)
            finally:
                task_queue.task_done()

    worker = threading.Thread(target=generate_worker, daemon=True)
    worker.start()

    # 推理只负责存结果, 评测由根目录的 eval.py 单独执行
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_f = open(args.out, 'w', encoding='utf-8')
    done_num = 0

    try:
        for inputs in dataset_val:
            messages = inputs["messages"]
            video_path = get_video_path(messages)
            video_name = os.path.basename(video_path) if video_path else "unknown"

            gt_answer = messages[-1]["content"][0]["text"]
            indx += 1
            if indx > max_samples:
                break

            model_inputs = new_collate_fn([inputs])
            model_inputs = model_inputs.to(model.device).to(model.dtype)

            task_queue.put(model_inputs)
            try:
                text_ids = result_queue.get(timeout=generate_timeout)
            except queue.Empty:
                print(f"!!! TIMEOUT for sample {indx} (>{generate_timeout}s), skipping (waiting for generate to finish)...")
                task_queue.join()
                continue

            if isinstance(text_ids, Exception):
                print(f"!!! Generate error for sample {indx}: {text_ids}")
                continue

            model_output = processor.decode(
                text_ids[0][model_inputs.input_ids.size(1):],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            done_num += 1
            print("+++" * 60)
            print("pred:", done_num, max_samples, model_output)
            print("g  t:", gt_answer)
            print("+++" * 60)

            out_f.write(json.dumps({
                "video_name": video_name,
                "pred": extract_answer(model_output),
                "gt": extract_answer(gt_answer),
            }, ensure_ascii=False) + '\n')
            out_f.flush()
    finally:
        out_f.close()
    print(f"共保存 {done_num} 条结果到 {args.out}, 评测请用: python eval.py {args.out} -t <task>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm-weights", default=os.path.join(BASE_DIR, "HumanOmniSpeaker", "llm_weights.pt"),
        help="Qwen2.5-Omni Thinker 主干权重 (由 convert_ckpt.py 生成)",
    )
    parser.add_argument(
        "--delta-weights", default=os.path.join(BASE_DIR, "HumanOmniSpeaker", "delta_encoder.pt"),
        help="DeltaEncoder 权重 (由 convert_ckpt.py 生成)",
    )
    parser.add_argument(
        "--dataset",
        default="/mnt/workspace/detao.bdt/datasets/human_omni_v25_lrs3_val.jsonl",
    )
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "results", "infer.jsonl"))
    parser.add_argument("--max-samples", type=int, default=0, help="最多推理样本数, 0 表示全部")
    parser.add_argument("--timeout", type=int, default=20, help="单样本生成超时秒数")
    parser.add_argument(
        "--use-delta-token", action=argparse.BooleanOptionalAction, default=True,
        help="是否使用 DeltaEncoder 的 token (填入 audio 占位符)",
    )
    parser.add_argument(
        "--use-vb-token", action=argparse.BooleanOptionalAction, default=False,
        help="是否使用原始 Omni 的 ViT 视觉 token",
    )
    parser.add_argument(
        "--use-audio-token", action=argparse.BooleanOptionalAction, default=False,
        help="是否使用原始 Omni 的 whisper 音频 token",
    )
    args = parser.parse_args()
    main(args)
