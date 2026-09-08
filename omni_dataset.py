import json
import math
import os
from typing import List, Union

import torch
from torch.utils.data import Dataset
from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniProcessorKwargs,
)
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import AudioInput, PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput, make_batched_videos

from qwen_omni_utils import process_mm_info


class ScriptArgs:
    def __init__(self, script_args_dict):
        self.__dict__.update(script_args_dict)


class OmniSpeakeProcessor(Qwen2_5OmniProcessor):

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[Qwen2_5OmniProcessorKwargs],
    ) -> BatchFeature:
        if text is None:
            raise ValueError("You need to specify either a `text` input to process.")

        output_kwargs = self._merge_kwargs(
            Qwen2_5OmniProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        seconds_per_chunk = output_kwargs["videos_kwargs"].pop("seconds_per_chunk")
        position_id_per_seconds = output_kwargs["videos_kwargs"].pop("position_id_per_seconds")
        use_audio_in_video = output_kwargs["videos_kwargs"].pop("use_audio_in_video")
        fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
        if "use_delta_token" in kwargs:
            use_delta_token = kwargs["use_delta_token"]
        else:
            use_delta_token = True

        if audio is not None:
            output_kwargs["audio_kwargs"]["padding"] = "max_length"  # Support "max_length" padding only here
            audio_inputs = self.feature_extractor(audio, **output_kwargs["audio_kwargs"])
            audio_inputs["feature_attention_mask"] = audio_inputs.pop(
                "attention_mask"
            )  # rename feature_attention_mask to prevent conflicts later on
            audio_inputs["input_features"] = audio_inputs.pop(
                "input_features"
            )  # rename input_features to prevent conflicts later on
            input_lengths = (audio_inputs["feature_attention_mask"].sum(-1) - 1) // 2 + 1
            if videos is not None and use_delta_token:
                # 根据audio,新增一份input_lengths
                new_input_lengths = []
                for tt, audio_length in zip(text, input_lengths):
                    count = tt.count("<|audio_bos|><|AUDIO|><|audio_eos|>")
                    if videos is not None:
                        visual_token_num = 1 if (videos[0].shape[-1] * videos[0].shape[-2]) / (224 * 224) < 1.3 else 6
                    else:
                        visual_token_num = 0
                    if visual_token_num > 1:
                        new_input_lengths.extend([torch.tensor(math.floor(audio_length / 2) * 2 * visual_token_num)])
                    else:
                        new_input_lengths.extend([audio_length])
                    if count > 1:
                        new_input_lengths.extend([audio_length] * (count - 1))

                input_lengths = torch.tensor(new_input_lengths)

            audio_lengths = iter((input_lengths - 2) // 2 + 1)
        else:
            input_lengths = []
            audio_inputs = {}
            audio_lengths = iter([])

        if images is not None:
            images_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = iter(images_inputs["image_grid_thw"])
        else:
            images_inputs = {}
            image_grid_thw = iter([])

        if videos is not None:
            videos = make_batched_videos(videos)
            videos_inputs = self.video_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            fps = [fps] * len(videos)
            videos_inputs["video_second_per_grid"] = [
                self.video_processor.temporal_patch_size / fps[i] for i in range(len(fps))
            ]
            video_grid_thw = iter(videos_inputs["video_grid_thw"])
            video_second_per_grid = iter(videos_inputs["video_second_per_grid"])
        else:
            videos_inputs = {}
            video_grid_thw = iter([])
            video_second_per_grid = iter([])

        if not isinstance(text, list):
            text = [text]

        text = self.replace_multimodal_special_tokens(
            text,
            audio_lengths,
            image_grid_thw,
            video_grid_thw,
            video_second_per_grid=video_second_per_grid,
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=position_id_per_seconds,
            seconds_per_chunk=seconds_per_chunk,
        )

        texts_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(
            data={**texts_inputs, **images_inputs, **videos_inputs, **audio_inputs},
            tensor_type=kwargs.get("return_tensors"),
        )


processor = OmniSpeakeProcessor.from_pretrained(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "HumanOmniSpeaker"),
    trust_remote_code=True,
)


class OmniSpeakerDataset(Dataset):
    def __init__(self, data_path: str, script_args):
        super(OmniSpeakerDataset, self).__init__()
        self.script_args = script_args

        print(data_path)

        with open(data_path, 'r', encoding='utf-8') as f:
            self.list_data_dict = [json.loads(line) for line in f if line.strip()]
        print(f"dataset_length:{len(self.list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, index):
        return self.load_msg(index)

    def load_msg(self, i):
        anno = self.list_data_dict[i]
        messages = anno["messages"]
        max_duration = anno["max_duration"]

        if "max_pixels" in messages[1]["content"][-1]:
            messages[1]["content"][-1]["fps"] = 25
            messages[1]["content"][-1]["max_pixels"] = 235200

        audios, images, videos, video_kwargs = process_mm_info(messages, use_audio_in_video=False, return_video_kwargs=True)
        assert audios is not None, print(f"+++++++音频数据不能为空++++++++++,{messages}")
        assert audios[0].shape[0] > 3000, print(f"+++++++音频必须要>100++++++++++{messages}")

        if videos is not None:
            assert videos[0].shape[0] > 5, print("+++++++只保留大于1s的视频数据++++++++++")
            assert videos[0].shape[0] < self.script_args.video_fps * max_duration, print(f"+++++++只保留{max_duration} s {videos[0].shape[0]}以内的视频数据++++++++++")

        return {
            'images': images,
            'audios': audios,
            'videos': videos,
            'messages': messages,
            'use_delta_token': self.script_args.use_delta_token,
            'use_vb_token': self.script_args.use_vb_token,
            'use_audio_token': self.script_args.use_audio_token,
            'dataset_name': anno["dataset_name"],
            'video_kwargs': video_kwargs,
        }


def sample_video_frames(video, video_fps, visual_slow_fps, factor=2):
    """
    对视频帧进行均匀采样，使输出帧率接近 visual_slow_fps，并保证采样帧数是 factor 的整数倍。
    """
    total_frames = video.shape[0]
    start_frame, end_frame = 0, total_frames - 1
    # 计算期望采样总帧数：原始时长 * 目标帧率
    nframes = max(2, int(total_frames / video_fps * visual_slow_fps))
    # 将帧数向下取整为 factor 的整数倍
    nframes = math.floor(nframes / factor) * factor

    indices = torch.linspace(start_frame, end_frame, nframes).round().long()
    idx_list = indices.tolist()
    sampled_fps = nframes / max(total_frames, 1e-6) * video_fps

    idx_list = [int(x * video_fps / 25) for x in idx_list]  # 归一化到25 fps对应的帧号

    new_video = video[indices]

    return new_video, sampled_fps, idx_list


class CustomCollater:
    def __init__(self, min_pixels=224 * 224, max_pixels=224 * 224, fps=25, mode="test"):
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.fps = fps
        self.visual_slow_fps = 2
        self.mode = mode

    def __call__(self, examples):
        images, videos, audios, prompts, video_sample_fps = [], [], [], [], []
        for each in examples:
            if self.mode == "test":
                prompts.append(each["messages"][:-1])
            else:
                prompts.append(each["messages"])

            if each["images"] is not None:
                images.extend(each["images"])
            if each["audios"] is not None:
                audios.extend(each["audios"])
            if each["videos"] is not None:
                videos.extend(each["videos"])
            if each["video_kwargs"] is not None:
                video_sample_fps.extend(each["video_kwargs"]["fps"])

        if len(images) == 0: images = None
        if len(audios) == 0: audios = None
        if len(videos) == 0: videos = None

        use_delta_token = examples[0]["use_delta_token"]
        use_vb_token = examples[0]["use_vb_token"]
        use_audio_token = examples[0]["use_audio_token"]
        dataset_name = examples[0]["dataset_name"]

        texts = processor.apply_chat_template(
            prompts,
            tokenize=False,
            add_generation_prompt=self.mode == "test",
        )

        # vb token: 不使用原始 ViT 视觉 token 时删除 video 占位符
        if not use_vb_token:
            for i in range(len(texts)):
                texts[i] = texts[i].replace("<|vision_bos|><|VIDEO|><|vision_eos|>", "")

        # audio 占位符: delta token 和 whisper token 都填入 audio 占位符位置
        if not use_delta_token and not use_audio_token:
            audios = None
            for i in range(len(texts)):
                texts[i] = texts[i].replace("<|audio_bos|><|AUDIO|><|audio_eos|>", "")
        elif use_delta_token and use_audio_token and videos is not None:
            # delta 特征与 whisper 特征各占一份占位符（仅在有视频产生 delta 特征时）
            for i in range(len(texts)):
                texts[i] = texts[i].replace("<|audio_bos|><|AUDIO|><|audio_eos|>", "<|audio_bos|><|AUDIO|><|audio_eos|>" * 2)

        print("texts:", texts)

        use_audio_in_video = False
        visual_slow_fps = self.visual_slow_fps if dataset_name not in ["Omni-DASR"] else 1

        # delta 路径需要的密集视频数据
        dense_videos_inputs = None
        if use_delta_token and videos is not None:
            video_kwargs = {'min_pixels': self.min_pixels, 'max_pixels': self.max_pixels, 'return_tensors': 'pt'}
            dense_videos = make_batched_videos(videos)
            dense_videos_inputs = processor.video_processor(images=None, videos=dense_videos, **video_kwargs)

        new_videos = None
        new_sampled_fps = [25]
        if videos is not None:
            new_videos = []
            new_sampled_fps = []
            for video, video_fps in zip(videos, video_sample_fps):
                new_video, sampled_fps, idx_list = sample_video_frames(video, video_fps, visual_slow_fps)
                new_videos.append(new_video)
                new_sampled_fps.append(sampled_fps)

        batch = processor(
            text=texts,
            images=images,
            audio=audios,
            videos=new_videos if (use_delta_token or use_vb_token) else None,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=use_audio_in_video,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            use_delta_token=use_delta_token,
            fps=new_sampled_fps[0],  # 只能传一个值
        )

        if dense_videos_inputs is not None:
            batch["dense_video_grid_thw"] = dense_videos_inputs["video_grid_thw"]
            batch["dense_pixel_values_videos"] = dense_videos_inputs["pixel_values_videos"]

        batch["use_audio_in_video"] = use_audio_in_video
        return batch
