"""
Qwen2.5-Omni Thinker 的自定义前向 (monkey-patch 用, 见 infer.py):
    Qwen2_5OmniThinkerForConditionalGeneration.forward = qwen_new_forward
    Qwen2_5OmniThinkerForConditionalGeneration.get_rope_index = get_rope_index

核心逻辑: 在 prefill 阶段把 delta_encoder 输出的稠密 AV 特征
(与原始 whisper 特征交错) 替换到 input_ids 的 audio 占位符位置。
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers.utils import ModelOutput


@dataclass
class Qwen2_5OmniThinkerCausalLMOutputWithPast(ModelOutput):
    """Qwen2.5OmniThinker causal LM output."""
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    encoder_hidden_states: Optional[torch.FloatTensor] = None


# ============================================================
# 音频 / 视频预处理（纯重排 + padding，无模型前向）
# ============================================================

def unpack_video_frames(pixel_values_videos, grid_thw, audio_output_lengths=None):
    """把 flatten 的视频 patch 恢复成逐帧图像 [T, 3, H, W]。

    并按 audio_output_lengths 截断/零填充每个视频的时间维, 与音频特征对齐。

    Returns:
        videos: [total_frames, 3, H, W] 所有视频帧按时间拼接
        video_lengths: [N] 每个视频的帧数
    """
    cu_seqlens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    merge_size, temporal_patch_size, channel, patch_size = 2, 2, 3, 14
    videos, video_lengths = [], []
    video_idx = 0
    for i, grid in enumerate(grid_thw):
        single_patch = pixel_values_videos[cu_seqlens[i]:cu_seqlens[i + 1]]
        single_patch = single_patch.view(
            -1, grid[0], grid[1] // merge_size, grid[2] // merge_size, merge_size, merge_size,
            channel, temporal_patch_size, patch_size, patch_size
        )
        single_patch = single_patch.permute(0, 1, 7, 6, 2, 4, 8, 3, 5, 9)
        single_patch = single_patch.reshape(
            single_patch.shape[0],
            grid[0] * temporal_patch_size,   # T
            channel,
            grid[1] * patch_size,            # H
            grid[2] * patch_size,            # W
        )
        for j in range(single_patch.shape[0]):
            video_feature = single_patch[j]
            if audio_output_lengths is not None:
                audio_len = audio_output_lengths[video_idx]
                if video_feature.shape[0] >= audio_len:
                    video_feature = video_feature[:audio_len]
                else:
                    pad = torch.zeros(
                        (audio_len - video_feature.shape[0],) + tuple(video_feature.shape[1:]),
                        device=video_feature.device, dtype=video_feature.dtype,
                    )
                    video_feature = torch.cat((video_feature, pad), dim=0)
            videos.extend(video_feature)
            video_lengths.append(video_feature.shape[0])
            video_idx += 1

    return torch.stack(videos), torch.tensor(video_lengths)


def _pad_video_chunks(chunk_list, chunk_lengths, max_len):
    """把 [len_i, C, H, W] 的视频 chunk pad 到 [N, max_len, C, H, W], 并生成有效帧 mask。"""
    c, h, w = chunk_list[0].shape[-3:]
    padded = torch.full(
        (len(chunk_list), max_len, c, h, w), 0,
        dtype=chunk_list[0].dtype, device=chunk_list[0].device,
    )
    mask = torch.arange(max_len, device=padded.device) < chunk_lengths.to(padded.device).unsqueeze(1)
    for i, length in enumerate(chunk_lengths):
        padded[i, :length] = chunk_list[i]
    return padded, mask


def _chunk_split(total_tensor, feature_lens, window):
    """按 window 把按样本拼接的时间维切成等长 chunk, 末尾 chunk 取余数长度。"""
    chunk_num = torch.ceil(feature_lens / window).long()
    chunk_lengths = torch.tensor(
        [window] * chunk_num.sum(), dtype=torch.long, device=feature_lens.device,
    )
    tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = feature_lens % window
    chunk_lengths = torch.where(chunk_lengths == 0, window, chunk_lengths)
    return total_tensor.split(chunk_lengths.tolist(), dim=0), chunk_lengths


# ============================================================
# delta_encoder 特征提取
# ============================================================

def prepare_video_embedding(delta_encoder, input_videos, feature_lens,
                            audio_output_lengths, audio_feat_lengths):
    """视频 chunk/pad 后过 delta_encoder, 产出 LLM token 特征。

    Returns:
        hidden_states: [total_valid_frames, hidden] 投影后的 LLM tokens
        encoder_hidden_states: [total_valid_frames, enc_dim] 投影前特征
    """
    # 窗口取三者最大值, 与音频长度对齐 (100 帧音频 == 25 帧视频, 故有 0.5)
    window = torch.max(
        torch.max(feature_lens.max(), audio_output_lengths.max()),
        torch.ceil(audio_feat_lengths.max() * 0.5),
    ).long().to(device=feature_lens.device)

    chunk_list, chunk_lengths = _chunk_split(input_videos, feature_lens, window)
    padded_feature, padded_mask_after_cnn = _pad_video_chunks(
        chunk_list, chunk_lengths, max_len=window
    )
    padded_feature = padded_feature.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

    padded_embed = delta_encoder.encoder(padded_feature, feature_lens)

    # encoder 输出时间维是窗口的整数倍; >1 时按 6 倍展开取有效帧
    num = int(padded_embed.shape[1] / padded_mask_after_cnn.shape[1])
    if num > 1:
        mask = (
            torch.arange(window * 6, device=padded_embed.device)
            < (chunk_lengths * 6).to(padded_embed.device).unsqueeze(1)
        )
        encoder_hidden_states = padded_embed[mask]
        hidden_states = delta_encoder.token_proj_6x(encoder_hidden_states)
    else:
        encoder_hidden_states = padded_embed[padded_mask_after_cnn]
        hidden_states = delta_encoder.token_proj_1x(encoder_hidden_states)

    return hidden_states, encoder_hidden_states


# ============================================================
# 特征回填 / 拼接 (qwen_new_forward 的构件)
# ============================================================

def _scatter_features(inputs_embeds, input_ids, token_id, features):
    """把 features 按占位符 token_id 回填到 inputs_embeds。"""
    mask = (
        (input_ids == token_id)
        .unsqueeze(-1)
        .expand_as(inputs_embeds)
        .to(inputs_embeds.device)
    )
    features = features.to(inputs_embeds.device, inputs_embeds.dtype)
    return inputs_embeds.masked_scatter(mask, features)


def _merge_delta_with_whisper(delta_features, whisper_features, audio_output_lengths):
    """多个音频占位符时, 把 delta 特征与原始 whisper 特征按段交错拼接。

    token_per_frame = delta 帧数 / whisper 帧数 (通常为 4)。
    Returns: (merged_features, token_per_frame)
    """
    token_per_frame = int(delta_features.shape[0] / whisper_features.shape[0])
    parts = []
    seq_idx, seq_idx_av = 0, 0
    for seq in audio_output_lengths:
        parts.append(delta_features[seq_idx_av:seq_idx_av + seq * token_per_frame])
        parts.append(whisper_features[seq_idx:seq_idx + seq])
        seq_idx += seq
        seq_idx_av += seq * token_per_frame
    return torch.cat(parts, dim=0), token_per_frame


# ============================================================
# monkey-patch: Thinker 前向
# ============================================================

def qwen_new_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        audio_feature_lengths: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_audio_in_video: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        video_second_per_grid: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, Qwen2_5OmniThinkerCausalLMOutputWithPast]:
        dense_video_grid_thw = kwargs.get("dense_video_grid_thw", None)
        dense_pixel_values_videos = kwargs.get("dense_pixel_values_videos", None)
        encoder_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extract the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

        audio_nums = torch.sum(input_ids[0] == self.config.audio_start_token_id)

        # 2. Merge text, audios, image and video
        token_per_frame = 1
        audio_output_lengths = None
        if input_ids is not None and input_ids.shape[1] != 1:  # Prefill stage
            if dense_pixel_values_videos is not None and self.delta_encoder is not None:
                # AV 分支: delta_encoder 特征 (+ 多占位符时与 whisper 交错)
                if feature_attention_mask is not None:
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                else:
                    audio_feature_lengths = None

                audio_feat_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
                )

                videos, video_lengths = unpack_video_frames(
                    dense_pixel_values_videos, dense_video_grid_thw, audio_output_lengths
                )

                # delta encoder 仅使用视觉特征
                delta_features, encoder_hidden_states = prepare_video_embedding(
                    self.delta_encoder, videos, video_lengths, audio_output_lengths, audio_feat_lengths
                )

                if audio_nums > 1:
                    whisper_features = self.get_audio_features(
                        input_features,
                        feature_attention_mask=feature_attention_mask,
                        audio_feature_lengths=audio_feature_lengths,
                    )
                    delta_features, token_per_frame = _merge_delta_with_whisper(
                        delta_features, whisper_features, audio_output_lengths
                    )

                inputs_embeds = _scatter_features(
                    inputs_embeds, input_ids, self.config.audio_token_id, delta_features
                )

            elif input_features is not None:
                # 纯音频分支: 原始 whisper 特征
                if feature_attention_mask is not None:
                    audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
                    _, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(audio_feature_lengths)
                audio_features = self.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                )
                inputs_embeds = _scatter_features(
                    inputs_embeds, input_ids, self.config.audio_token_id, audio_features
                )

            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                inputs_embeds = _scatter_features(
                    inputs_embeds, input_ids, self.config.image_token_id, image_embeds
                )

            if pixel_values_videos is not None:
                # 无/仅一个视频占位符时跳过 (与原实现一致)
                if (input_ids == self.config.video_token_id).sum() > 1:
                    video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                    inputs_embeds = _scatter_features(
                        inputs_embeds, input_ids, self.config.video_token_id, video_embeds
                    )

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # 3. 计算 rope 用的音频长度 (AV 分支时按 token_per_frame 展开)
        if audio_output_lengths is not None and audio_nums > 0:
            audio_feature_lengths = audio_output_lengths.repeat_interleave(audio_nums)
            audio_feature_lengths[0] = token_per_frame * audio_feature_lengths[0]
        else:
            audio_feature_lengths = None

        # 4. position_ids
        if attention_mask is not None and position_ids is None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask,
                    use_audio_in_video,
                    audio_feature_lengths,
                    video_second_per_grid,
                )
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                batch_size, seq_length = input_ids.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.get_text_config().vocab_size
            )

        if not return_dict:
            output = (logits,) + outputs
            return (loss,) + output if loss is not None else output

        return Qwen2_5OmniThinkerCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
            encoder_hidden_states=encoder_hidden_states,
        )


# ============================================================
# monkey-patch: 3D rope index
# ============================================================

def get_rope_index(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    use_audio_in_video: bool = False,
    audio_seqlens: Optional[torch.LongTensor] = None,
    second_per_grids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 3D rope position_ids (改写自 transformers 官方实现)。

    与官方区别: audio 段直接使用 audio_seqlens 作为长度
    (AV 分支时已按 token_per_frame 展开), 不再做 //2+1 的下采样换算。
    """
    spatial_merge_size = self.spatial_merge_size
    image_token_id = self.config.image_token_id
    video_token_id = self.config.video_token_id
    audio_token_id = self.config.audio_token_id
    vision_start_token_id = self.config.vision_start_token_id
    audio_start_token_id = self.config.audio_start_token_id
    position_id_per_seconds = self.config.position_id_per_seconds
    seconds_per_chunk = self.config.seconds_per_chunk

    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_idx, video_idx, audio_idx = 0, 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = torch.sum(input_ids == audio_start_token_id)
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (
                (vision_tokens == audio_start_token_id).sum()
                if use_audio_in_video
                else (vision_tokens == video_token_id).sum()
            )
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums
            multimodal_nums = (
                image_nums + audio_nums if use_audio_in_video else image_nums + video_nums + audio_nums
            )
            for _ in range(multimodal_nums):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                min_ed = min(ed_image, ed_video, ed_audio)
                if min_ed == ed_audio:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = audio_seqlens[audio_idx]
                    llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                    st += text_len + bos_len + audio_len + eos_len

                    audio_idx += 1
                    remain_audios -= 1
                elif min_ed == ed_image:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = image_grid_thw[image_idx][0]
                    grid_hs = image_grid_thw[:, 1]
                    grid_ws = image_grid_thw[:, 2]
                    t_index = (torch.arange(grid_t) * 1 * position_id_per_seconds).long()
                    llm_pos_ids = self.get_llm_pos_ids_for_vision(
                        st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + image_len + eos_len
                    image_idx += 1
                    remain_images -= 1

                elif min_ed == ed_video and not use_audio_in_video:
                    text_len = min_ed - st - 1
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]
                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).long()
                    llm_pos_ids = self.get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
                    llm_pos_ids_list.append(llm_pos_ids)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len + video_len + eos_len
                    video_idx += 1
                    remain_videos -= 1

                elif min_ed == ed_video and use_audio_in_video:
                    text_len = min_ed - st - 2
                    if text_len != 0:
                        st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                        llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    bos_len = 1
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    audio_len = audio_seqlens[audio_idx]
                    audio_llm_pos_ids = torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                    grid_t = video_grid_thw[video_idx][0]
                    grid_hs = video_grid_thw[:, 1]
                    grid_ws = video_grid_thw[:, 2]

                    t_index = (
                        torch.arange(grid_t) * second_per_grids[video_idx].cpu().float() * position_id_per_seconds
                    ).long()
                    video_llm_pos_ids = self.get_llm_pos_ids_for_vision(
                        st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
                    )

                    t_ntoken_per_chunk = int(position_id_per_seconds * seconds_per_chunk)
                    video_chunk_indexes = self.get_chunked_index(video_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                    audio_chunk_indexes = self.get_chunked_index(audio_llm_pos_ids[0], t_ntoken_per_chunk, st_idx)
                    for j in range(max(len(video_chunk_indexes), len(audio_chunk_indexes))):
                        video_chunk_index = video_chunk_indexes[j] if j < len(video_chunk_indexes) else None
                        audio_chunk_index = audio_chunk_indexes[j] if j < len(audio_chunk_indexes) else None
                        if video_chunk_index is not None:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[:, video_chunk_index[0] : video_chunk_index[1]]
                            )
                        if audio_chunk_index is not None:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[:, audio_chunk_index[0] : audio_chunk_index[1]]
                            )
                    video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    eos_len = 1
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)
                    llm_pos_ids_list.append(torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx)

                    st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2

                    audio_idx += 1
                    video_idx += 1
                    remain_videos -= 1
                    remain_audios -= 1

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)

            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)

        return position_ids, mrope_position_deltas
    else:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
        max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        mrope_position_deltas = max_position_ids + 1 - torch.sum(attention_mask, dim=-1, keepdim=True)

        return position_ids, mrope_position_deltas


# ============================================================
# 猴子补丁入口 (infer.py 调用)
# ============================================================

def apply_forward_patch():
    """打猴子补丁:
    1. Thinker.forward / get_rope_index 替换为本文件的自定义实现
    2. 关闭 GenerationMixin 的 kwargs 校验 (generate 需透传
       dense_pixel_values_videos 等自定义字段)
    """
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration
    from transformers.generation.utils import GenerationMixin

    def _nop_validate(self, *args, **kwargs):
        pass

    GenerationMixin._validate_model_kwargs = _nop_validate
    GenerationMixin._validate_generation_mode = _nop_validate

    Qwen2_5OmniThinkerForConditionalGeneration.forward = qwen_new_forward
    Qwen2_5OmniThinkerForConditionalGeneration.get_rope_index = get_rope_index
