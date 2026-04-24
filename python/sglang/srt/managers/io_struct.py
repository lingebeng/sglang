# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
不同进程（TokenizerManager、DetokenizerManager、Scheduler）之间传递的对象定义。
"""

"""
这个文件定义了 SGLang 各进程间传递的数据结构（Data Transfer Objects），是理解 SGLang 架构的关键入口，

因为它清晰地展示了系统的三大核心组件之间的通信协议：

TokenizerManager  ←→  Scheduler  ←→  DetokenizerManager
"""


"""
文件结构一览

  ┌─────────────────────────────────────────────────────────┐
  │  基类 (L51-90)                                           │
  │  ├── BaseReq          单条请求/响应基类                    │
  │  └── BaseBatchReq     批量请求/响应基类                    │
  ├─────────────────────────────────────────────────────────┤
  │  核心请求 (L141-1060) ← 最重要，占了一半篇幅                 │
  │  ├── GenerateReqInput          用户生成请求（最复杂）       │
  │  ├── TokenizedGenerateReqInput  tokenize 后的生成请求     │
  │  ├── EmbeddingReqInput         Embedding 请求            │
  │  └── TokenizedEmbeddingReqInput tokenize 后的 Embedding  │
  ├─────────────────────────────────────────────────────────┤
  │  核心输出 (L1063-1220)                                   │
  │  ├── BatchTokenIDOutput    Scheduler → Detokenizer       │
  │  ├── BatchStrOutput        Detokenizer → TokenizerManager│
  │  └── BatchEmbeddingOutput  Embedding 结果                 │
  ├─────────────────────────────────────────────────────────┤
  │  控制面请求 (L1222-1880) ← 运维管理用，用到再看               │
  │  ├── 缓存管理    FlushCache / ClearHiCache / AttachHiCache│
  │  ├── 权重更新    UpdateWeightFromDisk/Distributed/Tensor  │
  │  ├── LoRA 管理   LoadLoRA / UnloadLoRA                   │
  │  ├── 调度控制    PauseGeneration / ContinueGeneration     │
  │  ├── Profiling  ProfileReq                                 │
  │  └── 负载查询    GetLoadsReqInput/Output                  │
"""

from __future__ import annotations

import copy
import uuid
from abc import ABC
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import torch

from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.embed_types import PositionalEmbeds
from sglang.srt.managers.schedule_batch import BaseFinishReason, Modality
from sglang.srt.multimodal.mm_utils import has_valid_data
from sglang.srt.observability.req_time_stats import (
    APIServerReqTimeStats,
    DPControllerReqTimeStats,
    SchedulerReqTimeStats,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.utils import ImageData

# 处理 Image 的 pydantic 序列化
if TYPE_CHECKING:
    from PIL.Image import Image
else:
    Image = Any


# BaseReq 和 BaseBatchReq 是 SGLang 所有请求/响应数据结构的基类
@dataclass
class BaseReq(ABC):  # @haifeng 这个是用户请求
    # 请求 ID（Request ID），用于全局追踪一个请求。可以是单个字符串，也可以是字符串列表（批量场景）
    rid: Optional[Union[str, List[str]]] = field(default=None, kw_only=True)
    # @haifeng http_worker_ipc：HTTP worker 的 IPC 地址，用于将结果回传给对应的 HTTP worker
    http_worker_ipc: Optional[str] = field(default=None, kw_only=True)

    # @haifeng IPC是指进程间通信（Inter-Process Communication），在 SGLang 中用于不同进程之间传递数据和消息。HTTP worker 是处理 HTTP 请求的进程，而其他组件（如 TokenizerManager、Scheduler、DetokenizerManager）可能在不同的进程中运行。通过 IPC，系统可以将请求的结果从处理进程发送回 HTTP worker，以便最终返回给客户端。
    def regenerate_rid(self):
        """生成新的请求 ID 并返回。"""
        if isinstance(self.rid, list):
            self.rid = [uuid.uuid4().hex for _ in range(len(self.rid))]
        else:
            self.rid = uuid.uuid4().hex
        return self.rid

    def _validate_rid_uniqueness(self):
        """验证批次内的请求 ID 是否唯一。"""
        if isinstance(self.rid, list) and len(set(self.rid)) != len(self.rid):
            counts = Counter(self.rid)
            duplicates = [rid for rid, count in counts.items() if count > 1]
            raise ValueError(
                f"Duplicate request IDs detected within the request: {duplicates}"
            )


@dataclass
class BaseBatchReq(ABC):  # @haifeng 这个是 系统内部的，一定是列表且一定不会重复
    rids: Optional[List[str]] = field(default=None, kw_only=True)
    http_worker_ipcs: Optional[List[str]] = field(default=None, kw_only=True)

    def regenerate_rids(self):
        """生成新的请求 ID 列表并返回。"""
        self.rids = [uuid.uuid4().hex for _ in range(len(self.rids))]
        return self.rids


@dataclass
class SpeculativeDecodingMetricsMixin:
    """
    包含推测解码指标的混入类。

    该类整合了推测解码指标，这些指标在支持推测解码的批量输出类型之间共享，以避免代码重复。
    """

    # 验证次数：验证前向传播的次数
    spec_verify_ct: List[int]

    # 接受的 token 数：推测解码过程中接受的 token 数量
    spec_accepted_tokens: List[int]

    # 接受直方图：列表的列表，每个内部列表表示直方图计数。
    # 列表索引 = 单步中接受的 token 数，列表值 = 具有该数量接受 token 的步数。
    # 例如：histogram[0] = 5 表示有 5 步接受了 0 个 token，histogram[3] = 10 表示有 10 步接受了 3 个 token。
    # 当推测解码被禁用时为空列表 []。
    spec_acceptance_histogram: List[List[int]]


# 会话参数
@dataclass
class SessionParams:
    id: Optional[str] = None
    rid: Optional[str] = None
    offset: Optional[int] = None
    replace: Optional[bool] = None
    drop_previous_output: Optional[bool] = None


"""
@haifeng 使用场景
  多轮对话时，前一轮的 KV cache 可以复用，不需要重新计算：
  第1轮: "你好" → 计算 KV cache，保存在 session 中
  第2轮: "今天天气怎么样" → 复用第1轮的 KV cache，从 offset 位置继续
  - offset — 告诉系统从 KV cache 的哪个 token 位置接着算，避免重复计算前面的内容
  - replace — 如果用户修改了对话中间的内容，设为 True 就会替换 offset 之后的 cache
  - drop_previous_output — 丢弃前一轮的生成输出，只保留输入部分的 cache
  简单来说
  Session 机制让多轮对话不用每次都从头算，SessionParams 就是控制怎么复用和管理之前缓存的参数。
"""


# 多模态输入数据的类型定义
# 每种模态的单个数据项类型
ImageDataInputItem = Union[Image, str, ImageData, Dict]
AudioDataInputItem = Union[str, Dict]
VideoDataInputItem = Union[str, Dict]
# 任意多模态数据项的联合类型
MultimodalDataInputItem = Union[
    ImageDataInputItem, VideoDataInputItem, AudioDataInputItem
]
# 支持单个项、列表或嵌套列表的格式类型，用于批处理
MultimodalDataInputFormat = Union[
    List[List[MultimodalDataInputItem]],
    List[MultimodalDataInputItem],
    MultimodalDataInputItem,
]


@dataclass
class GenerateReqInput(BaseReq):
    # 输入提示。可以是单个提示或一批提示。
    text: Optional[Union[List[str], str]] = None
    # 文本的 token ID；可以指定 text 或 input_ids 其中之一
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # input_ids 的嵌入向量；可以指定 text、input_ids 或 input_embeds 其中之一。
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # 放置在特定 token 位置的嵌入覆盖。
    # 运行时类型：Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    # 类型标注为 Any 以避免 Pydantic/FastAPI schema 错误（PositionalEmbeds 包含 torch.Tensor）。
    positional_embed_overrides: Any = None
    # 图像输入。可以是图像实例、文件名、URL 或 base64 编码字符串。
    # 格式可以为：
    # - 单个请求的单张图像
    # - 图像列表（批次中每个请求一张）
    # - 图像列表的列表（每个请求多张图像）
    # 另见 python/sglang/srt/utils.py:load_image 了解更多细节。
    image_data: Optional[MultimodalDataInputFormat] = None
    # 视频输入。与图像数据类似，可以是文件名、URL 或 base64 编码字符串。
    video_data: Optional[MultimodalDataInputFormat] = None
    # 音频输入。与图像数据类似，可以是文件名、URL 或 base64 编码字符串。
    audio_data: Optional[MultimodalDataInputFormat] = None
    # 采样参数。见下方描述。
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # 是否返回 logprobs。
    return_logprob: Optional[Union[List[bool], bool]] = None
    # 如果返回 logprobs，从提示中的哪个位置开始返回 logprobs。
    # 默认值为 "-1"，表示只返回输出 token 的 logprobs。
    logprob_start_len: Optional[Union[List[int], int]] = None
    # 如果返回 logprobs，每个位置返回的 top logprobs 数量。
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # 如果返回 logprobs，需要返回 logprob 的 token ID。
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
    # 是否在返回的 logprobs 中对 token 进行反 token 化为文本。
    return_text_in_logprobs: bool = False
    # 是否流式输出。
    stream: bool = False
    # 是否记录此请求的指标（例如 health_generate 调用不记录指标）
    log_metrics: bool = True
    # 是否返回隐藏状态
    return_hidden_states: Union[List[bool], bool] = False
    # 是否返回捕获的路由专家
    return_routed_experts: bool = False
    # 从提示中的哪个位置开始返回路由专家。
    routed_experts_start_len: int = 0

    # 图像数据的模态 [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # 用于持续提示的会话信息
    session_params: Optional[Union[List[Dict], Dict]] = None

    # LoRA 适配器的路径
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    # LoRA 适配器的 uid，应由 tokenizer manager 初始化
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # 用于高级采样控制的自定义 logit 处理器。必须是
    # python/sglang/srt/sampling/custom_logit_processor.py 中 `CustomLogitProcessor` 的序列化实例。
    # 使用处理器的 `to_str()` 方法生成序列化字符串。
    custom_logit_processor: Optional[Union[List[Optional[str]], str]] = None

    # 用于分离式推理
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None
    bootstrap_pair_key: Optional[Union[List[str], str]] = None
    decode_tp_size: Optional[Union[List[Optional[int]], int]] = None

    # 要求请求进行推理（仅限混合推理模型）
    require_reasoning: bool = False

    # 用于 DP 路由 -- 外部路由器分配特定的 DP worker
    routed_dp_rank: Optional[int] = None
    # 用于 PD 分离 -- 提示解码端哪个预填充 DP worker 拥有 KV 缓存
    disagg_prefill_dp_rank: Optional[int] = None
    # 已弃用：请使用 routed_dp_rank 代替
    data_parallel_rank: Optional[int] = None

    # 用于后台响应（OpenAI responses API）
    background: bool = False

    # 用于跟踪请求的会话 ID
    conversation_id: Optional[str] = None

    # 请求的优先级
    priority: Optional[int] = None

    # 用于分类请求的额外键（例如 cache_salt）
    extra_key: Optional[Union[List[str], str]] = None

    # 用于 routing-key 调度策略的路由键
    routing_key: Optional[str] = None

    # 是否禁止记录此请求的日志（例如由于 ZDR）
    no_logs: bool = False

    # 用于自定义指标标签
    custom_labels: Optional[Dict[str, str]] = None

    # （内部）是否返回图像生成的字节数据
    return_bytes: bool = False

    # 是否返回熵
    return_entropy: bool = False

    # 通过 Engine.generate/async_generate 传播追踪上下文
    external_trace_header: Optional[Dict] = None
    received_time: Optional[float] = None

    # 用于 EPD 分离式推理
    need_wait_for_mm_inputs: Optional[bool] = None
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None

    # 多模态分块控制（扩展）
    max_dynamic_patch: Optional[int] = None
    min_dynamic_patch: Optional[int] = None
    image_max_dynamic_patch: Optional[int] = None
    video_max_dynamic_patch: Optional[int] = None

    # 用于多项评分的预计算分隔符索引。
    # 批次级别：List[List[int]]（每个请求一个）。经 __getitem__ 后：List[int]。
    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def normalize_batch_and_arguments(self):
        """
        标准化请求的批次大小和参数。

        该方法处理各种输入格式，确保所有参数根据输入正确格式化为单个值或批次。
        它还处理并行采样的扩展，并为未指定的参数设置默认值。

        Raises:
            ValueError: 如果输入未正确指定（例如 text、input_ids、input_embeds
                       都未提供或全部提供）
        """
        if self.data_parallel_rank is not None:
            import warnings

            warnings.warn(
                "'data_parallel_rank' is deprecated, use 'routed_dp_rank' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.routed_dp_rank is None:
                self.routed_dp_rank = self.data_parallel_rank
            self.data_parallel_rank = None

        self._validate_inputs()
        self._determine_batch_size()
        self._handle_parallel_sampling()

        if self.is_single:
            self._normalize_single_inputs()
        else:
            self._normalize_batch_inputs()

        self._validate_rid_uniqueness()

    def _validate_inputs(self):
        """验证输入配置是否有效。"""
        if (
            self.text is None and self.input_ids is None and self.input_embeds is None
        ) or (
            self.text is not None
            and self.input_ids is not None
            and self.input_embeds is not None
        ):
            raise ValueError(
                "Either text, input_ids or input_embeds should be provided."
            )

    def _determine_batch_size(self):
        """判断这是单个样本还是批次，并确定批次大小。"""
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if len(self.input_ids) == 0:
                raise ValueError("input_ids cannot be empty.")
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            if isinstance(self.input_embeds[0][0], float):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_embeds)

    def _handle_parallel_sampling(self):
        """处理并行采样参数，并在需要时调整批次大小。"""
        # 确定并行采样数量
        if self.sampling_params is None:
            self.parallel_sample_num = 1
            return
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list):
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            for sampling_params in self.sampling_params:
                if self.parallel_sample_num != sampling_params.get("n", 1):
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )

        # 如果使用并行采样且为单个样本，则转换为批次
        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]
            if self.input_embeds is not None:
                self.input_embeds = [self.input_embeds]

    def _normalize_single_inputs(self):
        """标准化单个样本的输入。"""
        if self.sampling_params is None:
            self.sampling_params = {}
        if self.rid is None:
            self.rid = uuid.uuid4().hex
        if self.return_logprob is None:
            self.return_logprob = False
        if self.logprob_start_len is None:
            self.logprob_start_len = -1
        if self.top_logprobs_num is None:
            self.top_logprobs_num = 0
        if not self.token_ids_logprob:  # 涵盖 None 和 [] 两种情况
            self.token_ids_logprob = None

    def _normalize_batch_inputs(self):
        """标准化批次样本的输入，包括并行采样的扩展。"""
        # 计算扩展后的批次大小
        if self.parallel_sample_num == 1:
            num = self.batch_size
        else:
            # 扩展 parallel_sample_num
            num = self.batch_size * self.parallel_sample_num

        # 根据类型扩展输入
        self._expand_inputs(num)
        self._normalize_rid(num)
        self._normalize_lora_paths(num)
        self._normalize_image_data(num)
        self._normalize_video_data(num)
        self._normalize_audio_data(num)
        self._normalize_sampling_params(num)
        self._normalize_logprob_params(num)
        self._normalize_custom_logit_processor(num)
        self._normalize_bootstrap_params(num)

    def _expand_inputs(self, num):
        """为并行采样扩展主要输入（text、input_ids、input_embeds）。"""
        if self.text is not None:
            if not isinstance(self.text, list):
                raise ValueError("Text should be a list for batch processing.")
            self.text = self.text * self.parallel_sample_num
        elif self.input_ids is not None:
            if not isinstance(self.input_ids, list) or not isinstance(
                self.input_ids[0], list
            ):
                raise ValueError(
                    "input_ids should be a list of lists for batch processing."
                )
            self.input_ids = self.input_ids * self.parallel_sample_num
        elif self.input_embeds is not None:
            if not isinstance(self.input_embeds, list):
                raise ValueError("input_embeds should be a list for batch processing.")
            self.input_embeds = self.input_embeds * self.parallel_sample_num

    def _normalize_lora_paths(self, num):
        """标准化批处理的 LoRA 路径。"""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                self.lora_path = self.lora_path * self.parallel_sample_num
            else:
                raise ValueError("lora_path should be a list or a string.")

    def _normalize_image_data(self, num):
        """标准化批处理的图像数据。"""
        if self.image_data is None:
            self.image_data = [None] * num
        elif not isinstance(self.image_data, list):
            # 单张图像，转换为单图像列表的列表
            self.image_data = [[self.image_data]] * num
            self.modalities = ["image"] * num
        elif isinstance(self.image_data, list):
            # 处理空列表的情况 - 视为没有图像
            if len(self.image_data) == 0:
                self.image_data = [None] * num
                return

            if len(self.image_data) != self.batch_size:
                raise ValueError(
                    "The length of image_data should be equal to the batch size."
                )

            self.modalities = []
            if len(self.image_data) > 0 and isinstance(self.image_data[0], list):
                # 已经是列表的列表，保持原样
                for i in range(len(self.image_data)):
                    if self.image_data[i] is None or self.image_data[i] == [None]:
                        self.modalities.append(None)
                    elif len(self.image_data[i]) == 1:
                        self.modalities.append("image")
                    elif len(self.image_data[i]) > 1:
                        self.modalities.append("multi-images")
                    else:
                        # 确保 len(self.modalities) == len(self.image_data)
                        self.modalities.append(None)
                # 扩展 parallel_sample_num
                self.image_data = self.image_data * self.parallel_sample_num
                self.modalities = self.modalities * self.parallel_sample_num
            else:
                # 批次的图像列表，将每个包装在列表中
                wrapped_images = [[img] for img in self.image_data]
                # 为并行采样扩展
                self.image_data = wrapped_images * self.parallel_sample_num
                self.modalities = ["image"] * num

    def _normalize_video_data(self, num):
        """标准化批处理的视频数据。"""
        if self.video_data is None:
            self.video_data = [None] * num
        elif not isinstance(self.video_data, list):
            self.video_data = [self.video_data] * num
        elif isinstance(self.video_data, list):
            self.video_data = self.video_data * self.parallel_sample_num

    def _normalize_audio_data(self, num):
        """标准化批处理的音频数据。"""
        if self.audio_data is None:
            self.audio_data = [None] * num
        elif not isinstance(self.audio_data, list):
            self.audio_data = [self.audio_data] * num
        elif isinstance(self.audio_data, list):
            self.audio_data = self.audio_data * self.parallel_sample_num

    def _normalize_sampling_params(self, num):
        """标准化批处理的采样参数。"""
        if self.sampling_params is None:
            self.sampling_params = [{}] * num
        elif isinstance(self.sampling_params, dict):
            self.sampling_params = [self.sampling_params] * num
        else:  # 已经是列表
            self.sampling_params = self.sampling_params * self.parallel_sample_num

    def _normalize_rid(self, num):
        """标准化批处理的请求 ID。"""
        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(num)]
        elif isinstance(self.rid, str):
            new_rids = [f"{self.rid}_{i}" for i in range(num)]
            self.rid = new_rids
        elif isinstance(self.rid, list):
            # 注意：rid 的长度应与 batch_size 相同，
            # 因为 rid 会在 tokenizer_manager 中为并行采样进行扩展
            if len(self.rid) != self.batch_size:
                raise ValueError(
                    "The specified rids length mismatch with the batch_size for batch processing."
                )
        else:
            raise ValueError("The rid should be a string or a list of strings.")

    def _normalize_logprob_params(self, num):
        """标准化批处理的 logprob 相关参数。"""

        # 标准化参数的辅助函数
        def normalize_param(param, default_value, param_name):
            if param is None:
                return [default_value] * num
            elif not isinstance(param, list):
                return [param] * num
            else:
                if self.parallel_sample_num > 1:
                    raise ValueError(
                        f"Cannot use list {param_name} with parallel_sample_num > 1"
                    )
                return param

        # 标准化每个 logprob 参数
        self.return_logprob = normalize_param(
            self.return_logprob, False, "return_logprob"
        )
        self.logprob_start_len = normalize_param(
            self.logprob_start_len, -1, "logprob_start_len"
        )
        self.top_logprobs_num = normalize_param(
            self.top_logprobs_num, 0, "top_logprobs_num"
        )

        # 由于嵌套结构，需要特殊处理 token_ids_logprob
        if not self.token_ids_logprob:  # 涵盖 None 和 [] 两种情况
            self.token_ids_logprob = [None] * num
        elif not isinstance(self.token_ids_logprob, list):
            self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
        elif not isinstance(self.token_ids_logprob[0], list):
            self.token_ids_logprob = [
                copy.deepcopy(self.token_ids_logprob) for _ in range(num)
            ]
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list token_ids_logprob with parallel_sample_num > 1"
            )

    def _normalize_custom_logit_processor(self, num):
        """标准化批处理的自定义 logit 处理器。"""
        if self.custom_logit_processor is None:
            self.custom_logit_processor = [None] * num
        elif not isinstance(self.custom_logit_processor, list):
            self.custom_logit_processor = [self.custom_logit_processor] * num
        elif self.parallel_sample_num > 1:
            raise ValueError(
                "Cannot use list custom_logit_processor with parallel_sample_num > 1"
            )

    def _normalize_bootstrap_params(self, num):
        """标准化批处理的 bootstrap 参数。"""
        # 标准化 bootstrap_host
        if self.bootstrap_host is None:
            self.bootstrap_host = [None] * num
        elif not isinstance(self.bootstrap_host, list):
            self.bootstrap_host = [self.bootstrap_host] * num
        elif isinstance(self.bootstrap_host, list):
            self.bootstrap_host = self.bootstrap_host * self.parallel_sample_num

        # 标准化 bootstrap_port
        if self.bootstrap_port is None:
            self.bootstrap_port = [None] * num
        elif not isinstance(self.bootstrap_port, list):
            self.bootstrap_port = [self.bootstrap_port] * num
        elif isinstance(self.bootstrap_port, list):
            self.bootstrap_port = self.bootstrap_port * self.parallel_sample_num

        # 标准化 bootstrap_room
        if self.bootstrap_room is None:
            self.bootstrap_room = [None] * num
        elif not isinstance(self.bootstrap_room, list):
            self.bootstrap_room = [self.bootstrap_room + i for i in range(num)]
        elif isinstance(self.bootstrap_room, list):
            self.bootstrap_room = self.bootstrap_room * self.parallel_sample_num

        # 标准化 bootstrap_pair_key
        if self.bootstrap_pair_key is None:
            self.bootstrap_pair_key = [None] * num
        elif not isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = [self.bootstrap_pair_key] * num
        elif isinstance(self.bootstrap_pair_key, list):
            self.bootstrap_pair_key = self.bootstrap_pair_key * self.parallel_sample_num

    def _validate_session_params(self):
        """验证会话参数的格式是否正确。"""
        if self.session_params is not None:
            if not isinstance(self.session_params, dict) and not isinstance(
                self.session_params[0], dict
            ):
                raise ValueError("Session params must be a dict or a list of dicts.")

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:
        """从 positional_embed_overrides 中提取第 i 个项。"""
        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):
        # 缓存子对象，确保重复调用 obj[i] 返回同一实例。
        # 这避免了不同调用点获取到不同对象的微妙 bug。
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]
        sub = GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            input_embeds=(
                self.input_embeds[i] if self.input_embeds is not None else None
            ),
            positional_embed_overrides=self._get_positional_embed_overrides_item(i),
            image_data=self.image_data[i],
            video_data=self.video_data[i],
            audio_data=self.audio_data[i],
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            log_metrics=self.log_metrics,
            return_hidden_states=(
                self.return_hidden_states[i]
                if isinstance(self.return_hidden_states, list)
                else self.return_hidden_states
            ),
            return_routed_experts=self.return_routed_experts,
            modalities=self.modalities[i] if self.modalities else None,
            session_params=self.session_params,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
            custom_logit_processor=(
                self.custom_logit_processor[i]
                if self.custom_logit_processor is not None
                else None
            ),
            # 如果调用了 `__getitem__`，bootstrap_host、bootstrap_port、bootstrap_room 必须是列表
            bootstrap_host=(
                self.bootstrap_host[i] if self.bootstrap_host is not None else None
            ),
            bootstrap_port=(
                self.bootstrap_port[i] if self.bootstrap_port is not None else None
            ),
            bootstrap_room=(
                self.bootstrap_room[i] if self.bootstrap_room is not None else None
            ),
            bootstrap_pair_key=(
                self.bootstrap_pair_key[i]
                if self.bootstrap_pair_key is not None
                else None
            ),
            decode_tp_size=(
                self.decode_tp_size[i] if self.decode_tp_size is not None else None
            ),
            routed_dp_rank=self.routed_dp_rank,
            disagg_prefill_dp_rank=self.disagg_prefill_dp_rank,
            conversation_id=self.conversation_id,
            priority=self.priority,
            extra_key=self.extra_key,
            no_logs=self.no_logs,
            custom_labels=self.custom_labels,
            return_bytes=self.return_bytes,
            return_entropy=self.return_entropy,
            external_trace_header=self.external_trace_header,
            http_worker_ipc=self.http_worker_ipc,
            received_time=self.received_time,
            multi_item_delimiter_indices=(
                self.multi_item_delimiter_indices[i]
                if self.multi_item_delimiter_indices is not None
                else None
            ),
        )
        cache[i] = sub
        return sub


@dataclass
class TokenizedGenerateReqInput(BaseReq):
    # 输入文本
    input_text: str
    # 输入 token ID
    input_ids: List[int]
    # 多模态输入
    mm_inputs: object
    # 采样参数
    sampling_params: SamplingParams
    # 是否返回 logprobs
    return_logprob: bool
    # 如果返回 logprobs，从提示中的哪个位置开始返回 logprobs。
    logprob_start_len: int
    # 如果返回 logprobs，每个位置返回的 top logprobs 数量。
    top_logprobs_num: int
    # 如果返回 logprobs，需要返回 logprob 的 token ID
    token_ids_logprob: List[int]
    # 是否流式输出
    stream: bool

    # 是否返回隐藏状态
    return_hidden_states: bool = False

    # 是否返回捕获的路由专家
    return_routed_experts: bool = False
    # 从提示中的哪个位置开始返回路由专家。
    routed_experts_start_len: int = 0

    # 输入嵌入向量
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None

    # 放置在特定 token 位置的嵌入覆盖。
    positional_embed_overrides: Optional[PositionalEmbeds] = None

    # 用于持续提示的会话信息
    session_params: Optional[SessionParams] = None

    # LoRA 相关
    lora_id: Optional[str] = None  # None 表示仅使用基础模型

    # 用于高级采样控制的自定义 logit 处理器。必须是
    # python/sglang/srt/sampling/custom_logit_processor.py 中 `CustomLogitProcessor` 的序列化实例。
    # 使用处理器的 `to_str()` 方法生成序列化字符串。
    custom_logit_processor: Optional[str] = None

    # 用于分离式推理
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    bootstrap_room: Optional[int] = None
    bootstrap_pair_key: Optional[str] = None
    decode_tp_size: Optional[int] = None

    # 要求请求进行推理（仅限混合推理模型）
    require_reasoning: bool = False

    # 用于 DP 路由
    routed_dp_rank: Optional[int] = None
    # 用于 PD 分离 -- 提示解码端哪个预填充 DP worker 拥有 KV 缓存
    disagg_prefill_dp_rank: Optional[int] = None

    # 请求的优先级
    priority: Optional[int] = None

    # 用于分类请求的额外键（例如 cache_salt）
    extra_key: Optional[str] = None

    # 用于 routing-key 调度策略的路由键
    routing_key: Optional[str] = None

    # 是否禁止记录此请求的日志（例如由于 ZDR）
    no_logs: bool = False

    # （内部）是否返回图像生成的字节数据
    return_bytes: bool = False

    # 是否返回熵
    return_entropy: bool = False

    token_type_ids: Optional[List[int]] = None

    need_wait_for_mm_inputs: bool = False
    num_items_assigned: Optional[Dict[Modality, List[int]]] = None

    # 用于多项评分的预计算分隔符索引
    multi_item_delimiter_indices: Optional[List[int]] = None

    # 用于可观测性
    time_stats: Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] = None


@dataclass
class BatchTokenizedGenerateReqInput(BaseBatchReq):
    # token 化请求的批次
    batch: List[TokenizedGenerateReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class EmbeddingReqInput(BaseReq):
    # 输入提示。可以是单个提示或一批提示。
    text: Optional[Union[List[List[str]], List[str], str]] = None
    # 图像输入。可以是图像实例、文件名、URL 或 base64 编码字符串。
    # 格式可以为：
    # - 单个请求的单张图像
    # - 图像列表（批次中每个请求一张）
    # - 图像列表的列表（每个请求多张图像）
    # 另见 python/sglang/srt/utils.py:load_image 了解更多细节。
    image_data: Optional[MultimodalDataInputFormat] = None
    # 视频输入。与图像数据类似，可以是文件名、URL 或 base64 编码字符串。
    video_data: Optional[MultimodalDataInputFormat] = None
    # 音频输入。与图像数据类似，可以是文件名、URL 或 base64 编码字符串。
    audio_data: Optional[MultimodalDataInputFormat] = None
    # 文本的 token ID；可以指定 text 或 input_ids 其中之一。
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # 用于定位输入 token ID 中嵌入覆盖位置的占位符 token ID。
    embed_override_token_id: Optional[int] = None
    # 未解析的嵌入覆盖：每个输入的张量列表。
    # 位置解析在 tokenizer manager 中的 token 化之后进行。
    # 形状：[num_inputs][num_replacements]，每个条目是 [hidden_size] 的 torch.Tensor。
    # 当批次中只有部分输入需要覆盖时，每个输入的条目可能为 None。
    # 运行时类型：Optional[List[Optional[List[torch.Tensor]]]]
    # 类型标注为 Any 以避免 Pydantic/FastAPI schema 错误（包含 torch.Tensor）。
    embed_overrides: Any = None
    # 已解析的带位置信息的嵌入覆盖（由 tokenizer manager 或 score mixin 设置）。
    # 运行时类型：Optional[Union[PositionalEmbeds, List[Optional[PositionalEmbeds]]]]
    positional_embed_overrides: Any = None
    # 用于兼容性的虚拟采样参数
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # 用于兼容性的虚拟输入嵌入
    input_embeds: Optional[Union[List[List[List[float]]], List[List[float]]]] = None
    # 是否记录此请求的指标（例如 health_generate 调用不记录指标）
    log_metrics: bool = True
    # 图像数据的模态 [image, multi-images, video]
    modalities: Optional[List[str]] = None
    # 用于交叉编码器请求
    is_cross_encoder_request: bool = False
    # 请求的优先级
    priority: Optional[int] = None
    # 用于 routing-key 调度策略的路由键
    routing_key: Optional[str] = None

    # 用于后台响应（OpenAI responses API）
    background: bool = False

    # 通过 Engine.encode/async_encode 传播追踪上下文
    external_trace_header: Optional[Dict] = None
    received_time: Optional[float] = None

    # 结果输出嵌入应具有的维度数。适用于 Matryoshka 嵌入。
    dimensions: Optional[int] = None

    # LoRA 适配器的路径
    lora_path: Optional[Union[List[Optional[str]], Optional[str]]] = None
    # LoRA 适配器的 uid，应由 tokenizer manager 初始化
    lora_id: Optional[Union[List[Optional[str]], Optional[str]]] = None

    # 是否返回池化后的隐藏状态（head 之前的 transformer 输出）
    return_pooled_hidden_states: bool = False

    # 用于多项评分的预计算分隔符索引。
    # 批次级别：List[List[int]]（每个请求一个）。经 __getitem__ 后：List[int]。
    multi_item_delimiter_indices: Optional[Union[List[List[int]], List[int]]] = None

    def normalize_batch_and_arguments(self):
        # text、input_ids 或 image 至少需要提供一个
        if self.text is None and self.input_ids is None and self.image_data is None:
            raise ValueError(
                "At least one of text, input_ids, or image should be provided"
            )

        # text 和 input_ids 不能同时提供
        if self.text is not None and self.input_ids is not None:
            raise ValueError("text and input_ids cannot be provided at the same time")

        # 推导批次大小
        self.batch_size = 0
        self.is_single = True

        # 检查 text 的批次大小
        if self.text is not None:
            if isinstance(self.text, list):
                self.batch_size += len(self.text)
                self.is_single = False
            else:
                self.batch_size += 1

        # 检查 input_ids 的批次大小
        if self.input_ids is not None:
            if isinstance(self.input_ids[0], list):
                self.batch_size += len(self.input_ids)
                self.is_single = False
            else:
                self.batch_size += 1

        # 填充默认参数
        if self.is_single:
            if self.rid is None:
                self.rid = uuid.uuid4().hex
            if self.sampling_params is None:
                self.sampling_params = {}
            self.sampling_params["max_new_tokens"] = 0
        else:
            if self.rid is None:
                self.rid = [uuid.uuid4().hex for _ in range(self.batch_size)]
            else:
                assert isinstance(self.rid, list), "The rid should be a list."

            if self.sampling_params is None:
                self.sampling_params = [{}] * self.batch_size
            elif isinstance(self.sampling_params, dict):
                self.sampling_params = [self.sampling_params] * self.batch_size
            for i in range(self.batch_size):
                self.sampling_params[i]["max_new_tokens"] = 0

            self._normalize_lora_paths(self.batch_size)

        self._validate_rid_uniqueness()

    def _normalize_lora_paths(self, num):
        """标准化批处理的 LoRA 路径。"""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                if len(self.lora_path) != num:
                    raise ValueError(
                        f"lora_path list length ({len(self.lora_path)}) must match batch size ({num})"
                    )
            else:
                raise ValueError("lora_path should be a list or a string.")

    def contains_mm_input(self) -> bool:
        return (
            has_valid_data(self.image_data)
            or has_valid_data(self.video_data)
            or has_valid_data(self.audio_data)
        )

    def _get_positional_embed_overrides_item(
        self, i: int
    ) -> Optional[PositionalEmbeds]:
        """从 positional_embed_overrides 中提取第 i 个项。"""
        if self.positional_embed_overrides is None:
            return None
        if isinstance(self.positional_embed_overrides, PositionalEmbeds):
            return self.positional_embed_overrides
        return self.positional_embed_overrides[i]

    def __getitem__(self, i):
        # 缓存子对象，确保重复调用 obj[i] 返回同一实例。
        cache = self.__dict__.setdefault("_sub_obj_cache", {})
        if i in cache:
            return cache[i]

        if self.is_cross_encoder_request:
            sub = EmbeddingReqInput(
                text=[self.text[i]] if self.text is not None else None,
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                sampling_params=self.sampling_params[i],
                rid=self.rid[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                is_cross_encoder_request=True,
                http_worker_ipc=self.http_worker_ipc,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        else:
            sub = EmbeddingReqInput(
                text=self.text[i] if self.text is not None else None,
                input_ids=self.input_ids[i] if self.input_ids is not None else None,
                embed_override_token_id=self.embed_override_token_id,
                embed_overrides=(
                    self.embed_overrides[i]
                    if self.embed_overrides is not None
                    else None
                ),
                positional_embed_overrides=self._get_positional_embed_overrides_item(i),
                image_data=self.image_data[i] if self.image_data is not None else None,
                audio_data=self.audio_data[i] if self.audio_data is not None else None,
                video_data=self.video_data[i] if self.video_data is not None else None,
                sampling_params=self.sampling_params[i],
                rid=self.rid[i],
                lora_path=self.lora_path[i] if self.lora_path is not None else None,
                lora_id=self.lora_id[i] if self.lora_id is not None else None,
                external_trace_header=self.external_trace_header,
                dimensions=self.dimensions,
                http_worker_ipc=self.http_worker_ipc,
                received_time=self.received_time,
                return_pooled_hidden_states=self.return_pooled_hidden_states,
                multi_item_delimiter_indices=(
                    self.multi_item_delimiter_indices[i]
                    if self.multi_item_delimiter_indices is not None
                    else None
                ),
            )
        cache[i] = sub
        return sub


@dataclass
class TokenizedEmbeddingReqInput(BaseReq):
    # 输入文本
    input_text: str
    # 输入 token ID
    input_ids: List[int]
    # 图像输入
    image_inputs: dict
    # token 类型 ID
    token_type_ids: List[int]
    # 用于兼容性的虚拟采样参数
    sampling_params: SamplingParams
    # 放置在特定 token 位置的嵌入覆盖。
    positional_embed_overrides: Optional[PositionalEmbeds] = None
    # 用于 DP 路由
    routed_dp_rank: Optional[int] = None
    # 请求的优先级
    priority: Optional[int] = None
    # 结果输出嵌入应具有的维度数。适用于 Matryoshka 嵌入。
    dimensions: Optional[int] = None

    # LoRA 相关
    lora_id: Optional[str] = None  # None 表示仅使用基础模型
    # 用于多项评分的预计算分隔符索引
    multi_item_delimiter_indices: Optional[List[int]] = None
    # 用于可观测性
    time_stats: Optional[Union[APIServerReqTimeStats, DPControllerReqTimeStats]] = None

    # 是否返回池化后的隐藏状态（head 之前的 transformer 输出）
    return_pooled_hidden_states: bool = False


@dataclass
class BatchTokenizedEmbeddingReqInput(BaseBatchReq):
    # token 化嵌入请求的批次
    batch: List[TokenizedEmbeddingReqInput]

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, i):
        return self.batch[i]

    def __iter__(self):
        return iter(self.batch)


@dataclass
class BatchTokenIDOutput(BaseBatchReq, SpeculativeDecodingMetricsMixin):
    # 完成原因
    finished_reasons: List[BaseFinishReason]
    # 用于增量解码
    decoded_texts: List[str]
    decode_ids: List[int]
    read_offsets: List[int]
    # 仅在 `--skip-tokenizer-init` 开启时使用
    output_ids: Optional[List[int]]
    # 反 token 化配置
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]

    # token 计数
    prompt_tokens: List[int]
    reasoning_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    # 对数概率
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]
    output_token_entropy_val: List[float]

    # 隐藏状态
    output_hidden_states: List[List[float]]

    # 每个 token 的路由专家，包括输入和输出 token
    # routed_experts[i] 是形状为 (token, layer, top_k) 的张量，对应请求 i
    routed_experts: List[Optional[torch.Tensor]]

    # 占位符 token 的信息（例如图像 token）
    # idx 是扩展后 token 在提示中的索引。
    # val 是扩展后填充 token 的长度。
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    # 每个请求被回退的次数。
    retraction_counts: List[int]

    # 训练器步骤 ID。用于了解采样使用的是哪一步的权重。
    token_steps: List[List[int]] = None

    # 用于 DP 平衡的负载
    load: GetLoadsReqOutput = None
    # 自定义信息
    customized_info: Optional[Dict[str, List[Any]]] = None
    # 按来源（设备/主机/存储）分类的缓存 token 详细信息
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    # 处理每个请求的调度器的 DP 排名
    dp_ranks: Optional[List[int]] = None

    # 用于可观测性
    time_stats: Optional[List[SchedulerReqTimeStats]] = None


@dataclass
class BatchStrOutput(BaseBatchReq, SpeculativeDecodingMetricsMixin):
    # 完成原因
    finished_reasons: List[dict]
    # 输出解码后的字符串
    output_strs: List[str]
    # token ID
    output_ids: Optional[List[int]]

    # token 计数
    prompt_tokens: List[int]
    completion_tokens: List[int]
    reasoning_tokens: List[int]
    cached_tokens: List[int]

    # 对数概率
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]
    output_token_entropy_val: List[float]

    # 隐藏状态
    output_hidden_states: List[List[float]]

    # 每个 token 的路由专家，包括输入和输出 token
    # routed_experts[i] 是形状为 (token, layer, top_k) 的张量，对应请求 i
    routed_experts: List[Optional[torch.Tensor]]

    # 占位符 token 的信息（例如图像 token）
    # idx 是扩展后 token 在提示中的索引。
    # val 是扩展后填充 token 的长度。
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    # 每个请求被回退的次数。
    retraction_counts: List[int]

    # 训练器步骤 ID。用于了解采样使用的是哪一步的权重。
    token_steps: List[List[int]] = None

    # 用于 DP 平衡的负载
    load: GetLoadsReqOutput = None

    # 自定义信息
    customized_info: Optional[Dict[str, List[Any]]] = None
    # 按来源（设备/主机/存储）分类的缓存 token 详细信息
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None
    # 处理每个请求的调度器的 DP 排名
    dp_ranks: Optional[List[int]] = None

    # 用于可观测性
    time_stats: Optional[List[SchedulerReqTimeStats]] = None


@dataclass
class BatchEmbeddingOutput(BaseBatchReq):
    # 完成原因
    finished_reasons: List[BaseFinishReason]
    # 输出嵌入向量
    embeddings: Union[List[List[float]], List[Dict[int, float]]]
    # token 计数
    prompt_tokens: List[int]
    cached_tokens: List[int]
    # 占位符 token 信息
    placeholder_tokens_idx: List[Optional[List[int]]]
    placeholder_tokens_val: List[Optional[List[int]]]

    # 每个请求被回退的次数。
    retraction_counts: List[int]
    # 按来源（设备/主机/存储）分类的缓存 token 详细信息
    cached_tokens_details: Optional[List[Optional[Dict[str, Any]]]] = None

    # 用于可观测性
    time_stats: Optional[List[SchedulerReqTimeStats]] = None

    # 可选的池化后隐藏状态（head 之前的 transformer 输出）。
    # 作为单个堆叠张量发送，以减少 pickle 开销。
    pooled_hidden_states: Optional[
        Union[List[Optional[torch.Tensor]], torch.Tensor]
    ] = None


@dataclass
class ClearHiCacheReqInput(BaseReq):
    pass


@dataclass
class ClearHiCacheReqOutput(BaseReq):
    success: bool


@dataclass
class FlushCacheReqInput(BaseReq):
    timeout_s: Optional[float] = None


@dataclass
class FlushCacheReqOutput(BaseReq):
    success: bool
    message: str = ""


@dataclass
class AddExternalCorpusReqInput(BaseReq):
    corpus_id: Optional[str] = None
    file_path: Optional[str] = None
    documents: Optional[List[str]] = None
    token_chunks: Optional[List[List[int]]] = None


@dataclass
class AddExternalCorpusReqOutput(BaseReq):
    success: bool
    corpus_id: str = ""
    message: str = ""
    loaded_token_count: int = 0


@dataclass
class RemoveExternalCorpusReqInput(BaseReq):
    corpus_id: str


@dataclass
class RemoveExternalCorpusReqOutput(BaseReq):
    success: bool
    message: str = ""


@dataclass
class ListExternalCorporaReqInput(BaseReq):
    pass


@dataclass
class ListExternalCorporaReqOutput(BaseReq):
    success: bool
    corpus_token_counts: Dict[str, int] = field(default_factory=dict)
    message: str = ""


@dataclass
class AttachHiCacheStorageReqInput(BaseReq):
    """在运行时动态挂载（启用）HiCache 存储后端。

    注意：`hicache_storage_backend_extra_config_json` 是一个 JSON 字符串。它可能包含：
    - 后端特定的配置（例如 mooncake master 地址）
    - 预取相关的参数（prefetch_threshold、prefetch_timeout_*、hicache_storage_pass_prefix_keys）
    """

    hicache_storage_backend: str
    hicache_storage_backend_extra_config_json: Optional[str] = None
    hicache_storage_prefetch_policy: Optional[str] = None
    hicache_write_policy: Optional[str] = None

    def __post_init__(self):
        if self.hicache_storage_prefetch_policy is None:
            pass
        else:
            allowed = ["best_effort", "wait_complete", "timeout"]
            if self.hicache_storage_prefetch_policy not in allowed:
                raise ValueError(
                    f"Invalid hicache_storage_prefetch_policy: {self.hicache_storage_prefetch_policy!r}. "
                    f"Expected one of {allowed}."
                )

        if self.hicache_write_policy is None:
            return
        allowed = ["write_back", "write_through", "write_through_selective"]
        if self.hicache_write_policy not in allowed:
            raise ValueError(
                f"Invalid hicache_write_policy: {self.hicache_write_policy!r}. "
                f"Expected one of {allowed}."
            )


@dataclass
class AttachHiCacheStorageReqOutput(BaseReq):
    success: bool
    message: str = ""


@dataclass
class DetachHiCacheStorageReqInput(BaseReq):
    """在运行时动态卸载（禁用）HiCache 存储后端。"""

    pass


@dataclass
class DetachHiCacheStorageReqOutput(BaseReq):
    success: bool
    message: str = ""


@dataclass
class PauseGenerationReqInput(BaseReq):
    """
    注意 PauseGenerationRequests 仅在 SGLang Server 中支持。
    abort：终止并返回所有当前正在处理的请求。

    in_place：暂停调度器的 event_loop 执行推理；
            只处理非推理请求（例如控制命令）。
            引擎中的请求将被暂停并留在 event_loop 中，
            然后在 continue_generation 后使用旧的 KV 缓存继续生成。
            注意：在 'inplace' 模式下，如果 running_batch 中有任何请求，
            flush_cache 将会失败。

    retract：暂停调度器的 event_loop 执行推理；
            只处理非推理请求，所有当前运行的请求将被回退到 waiting_queue。
            注意：在此模式下可以刷新 KV 缓存，刷新后的 KV 缓存将在
            continue_generation 后自动重新计算。
    """

    mode: Literal["abort", "retract", "in_place"] = "abort"

    def __post_init__(self):
        allowed = ["abort", "retract", "in_place"]
        if self.mode not in allowed:
            raise ValueError(
                f"Invalid mode: {self.mode!r}. " f"Expected one of {allowed}."
            )


@dataclass
class ContinueGenerationReqInput(BaseReq):
    pass


@dataclass
class UpdateWeightFromDiskReqInput(BaseReq):
    # 包含新权重的模型路径
    model_path: str
    # 加载权重的格式
    load_format: Optional[str] = None
    # 是否在更新权重前终止所有请求
    abort_all_requests: bool = False
    # 可选：随权重一起更新权重版本
    weight_version: Optional[str] = None
    # 是否异步更新权重
    is_async: bool = False
    # 是否清空 torch 缓存
    torch_empty_cache: bool = False
    # 权重更新后是否保持调度器暂停
    keep_pause: bool = False
    # 权重更新后是否重新捕获 CUDA 计算图
    recapture_cuda_graph: bool = False
    # 训练器步骤 ID。用于了解采样使用的是哪一步的权重。
    token_step: int = 0
    # 更新权重后是否刷新缓存
    flush_cache: bool = True
    # 张量元数据
    manifest: Optional[Dict[str, Any]] = None


@dataclass
class UpdateWeightFromDiskReqOutput(BaseReq):
    success: bool
    message: str
    # 权重同步期间暂停的请求数量。
    num_paused_requests: Optional[int] = 0


@dataclass
class UpdateWeightsFromDistributedReqInput(BaseReq):
    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    # 通信组名称
    group_name: str = "weight_update_group"
    # 更新权重后是否刷新缓存
    flush_cache: bool = True
    # 是否在更新权重前终止所有请求
    abort_all_requests: bool = False
    # 可选：随权重一起更新权重版本
    weight_version: Optional[str] = None
    # 可选的加载格式规范
    load_format: Optional[str] = None


@dataclass
class UpdateWeightsFromDistributedReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class UpdateWeightsFromTensorReqInput(BaseReq):
    """从张量输入更新模型权重。

    - 张量经过序列化以进行传输
    - 数据以 JSON 格式组织，便于通过 HTTP 传输
    """

    serialized_named_tensors: List[Union[str, bytes]]
    # 可选的加载格式规范
    load_format: Optional[str] = None
    # 更新权重后是否刷新缓存
    flush_cache: bool = True
    # 是否在更新权重前终止所有请求
    abort_all_requests: bool = False
    # 可选：随权重一起更新权重版本
    weight_version: Optional[str] = None
    # 可选：是否禁用草稿模型的更新
    disable_draft_model: Optional[bool] = None


@dataclass
class UpdateWeightsFromTensorReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqInput(BaseReq):
    # 主节点地址
    master_address: str
    # 每个 rank 通信组的端口
    ports: str
    # 通信组中的 rank
    group_rank: int
    # 世界大小
    world_size: int
    # 通信组名称
    group_name: str = "weight_send_group"
    # 后端
    backend: str = "nccl"


# 目前 UpdateWeightsFromIPCReqInput 和 UpdateWeightsFromIPCReqOutput
# 仅被 Checkpoint Engine（https://github.com/MoonshotAI/checkpoint-engine）使用
@dataclass
class UpdateWeightsFromIPCReqInput(BaseReq):
    # 每个设备 UUID 的 ZMQ socket 路径
    zmq_handles: Dict[str, str]
    # 权重更新后是否刷新缓存
    flush_cache: bool = True
    # 可选：随权重一起更新权重版本
    weight_version: Optional[str] = None


@dataclass
class UpdateWeightsFromIPCReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class InitWeightsSendGroupForRemoteInstanceReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class SendWeightsToRemoteInstanceReqInput(BaseReq):
    # 主节点地址
    master_address: str
    # 每个 rank 通信组的端口
    ports: str
    # 通信组名称
    group_name: str = "weight_send_group"


@dataclass
class SendWeightsToRemoteInstanceReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class UpdateExpertBackupReq(BaseReq):
    pass


@dataclass
class BackupDramReq(BaseReq):
    rank: int
    weight_pointer_map: Dict[str, Any]
    session_id: str
    buffer_size: int


@dataclass
class InitWeightsUpdateGroupReqInput(BaseReq):
    # 主节点地址
    master_address: str
    # 主节点端口
    master_port: int
    # rank 偏移量
    rank_offset: int
    # 世界大小
    world_size: int
    # 通信组名称
    group_name: str = "weight_update_group"
    # 后端
    backend: str = "nccl"


@dataclass
class InitWeightsUpdateGroupReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class DestroyWeightsUpdateGroupReqInput(BaseReq):
    group_name: str = "weight_update_group"


@dataclass
class DestroyWeightsUpdateGroupReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class UpdateWeightVersionReqInput(BaseReq):
    # 新的权重版本
    new_version: str
    # 更新前是否终止所有正在运行的请求
    abort_all_requests: bool = True


@dataclass
class GetWeightsByNameReqInput(BaseReq):
    name: str
    truncate_size: int = 100


@dataclass
class GetWeightsByNameReqOutput(BaseReq):
    parameter: list


@dataclass
class ReleaseMemoryOccupationReqInput(BaseReq):
    # 用于标识内存区域的可选标签，主要用于 RL
    # 目前仅支持 `weights` 和 `kv_cache`
    tags: Optional[List[str]] = None


@dataclass
class ReleaseMemoryOccupationReqOutput(BaseReq):
    pass


@dataclass
class ResumeMemoryOccupationReqInput(BaseReq):
    # 用于标识内存区域的可选标签，主要用于 RL
    # 目前仅支持 `weights` 和 `kv_cache`
    tags: Optional[List[str]] = None


@dataclass
class ResumeMemoryOccupationReqOutput(BaseReq):
    pass


@dataclass
class CheckWeightsReqInput(BaseReq):
    action: str


@dataclass
class CheckWeightsReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class SlowDownReqInput(BaseReq):
    forward_sleep_time: Optional[float]


@dataclass
class SlowDownReqOutput(BaseReq):
    pass


@dataclass
class AbortReq(BaseReq):
    # 是否终止所有请求
    abort_all: bool = False
    # 完成原因数据
    finished_reason: Optional[Dict[str, Any]] = None
    abort_message: Optional[str] = None

    def __post_init__(self):
        # FIXME: 这是一个临时方案，用于保持与旧代码一致
        if self.rid is None:
            self.rid = ""


@dataclass
class ActiveRanksOutput(BaseReq):
    status: List[bool]


@dataclass
class GetInternalStateReq(BaseReq):
    pass


@dataclass
class GetInternalStateReqOutput(BaseReq):
    internal_state: Dict[Any, Any]


@dataclass
class SetInternalStateReq(BaseReq):
    server_args: Dict[str, Any]


@dataclass
class SetInternalStateReqOutput(BaseReq):
    updated: bool
    server_args: Dict[str, Any]


@dataclass
class ProfileReqInput(BaseReq):
    # 输出目录
    output_dir: Optional[str] = None
    # 指定开始性能分析的步骤
    start_step: Optional[int] = None
    # 如果设置，则分析该数量的步骤。
    # 如果设置了该参数，性能分析会在此步骤后自动停止，
    # 调用者不需要运行 stop_profile。
    num_steps: Optional[int] = None
    # 要记录的活动。选项为 ["CPU", "GPU", "MEM", "RPD"]
    activities: Optional[List[str]] = None
    # 是否按阶段（例如预填充和解码）分别进行性能分析
    profile_by_stage: bool = False
    # 是否记录操作的源信息（文件和行号）。
    with_stack: Optional[bool] = None
    # 是否保存操作符输入形状的信息。
    record_shapes: Optional[bool] = None
    # 合并所有 rank 的性能分析到单个 trace 中
    merge_profiles: bool = False
    # 性能分析文件名的前缀
    profile_prefix: Optional[str] = None
    # 仅分析这些阶段，忽略其他阶段
    profile_stages: Optional[List[str]] = None


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class ProfileReq(BaseReq):
    type: ProfileReqType
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    activities: Optional[List[str]] = None
    profile_by_stage: bool = False
    with_stack: Optional[bool] = None
    record_shapes: Optional[bool] = None
    profile_id: Optional[str] = None
    merge_profiles: bool = False
    profile_prefix: Optional[str] = None
    profile_stages: Optional[List[str]] = None


@dataclass
class ProfileReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class FreezeGCReq(BaseReq):
    pass


@dataclass
class ConfigureLoggingReq(BaseReq):
    log_requests: Optional[bool] = None
    log_requests_level: Optional[int] = None
    log_requests_format: Optional[str] = None
    dump_requests_folder: Optional[str] = None
    dump_requests_threshold: Optional[int] = None
    crash_dump_folder: Optional[str] = None


@dataclass
class OpenSessionReqInput(BaseReq):
    capacity_of_str_len: int
    session_id: Optional[str] = None
    streaming: Optional[bool] = None
    timeout: Optional[float] = None


@dataclass
class CloseSessionReqInput(BaseReq):
    session_id: str


@dataclass
class OpenSessionReqOutput(BaseReq):
    session_id: Optional[str]
    success: bool


@dataclass
class HealthCheckOutput(BaseReq):
    pass


class ExpertDistributionReqType(Enum):
    START_RECORD = 1
    STOP_RECORD = 2
    DUMP_RECORD = 3


@dataclass
class ExpertDistributionReq(BaseReq):
    action: ExpertDistributionReqType


@dataclass
class ExpertDistributionReqOutput(BaseReq):
    pass


@dataclass
class Function:
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Optional[object] = None


@dataclass
class Tool:
    function: Function
    type: Optional[str] = "function"


@dataclass
class ParseFunctionCallReq(BaseReq):
    text: str  # 要解析的文本。
    tools: List[Tool] = field(
        default_factory=list
    )  # 可用的函数工具列表（名称、参数等）。
    tool_call_parser: Optional[str] = (
        None  # 指定解析器类型，例如 'llama3'、'qwen25' 或 'mistral'。如果未指定，则尝试所有解析器。
    )


@dataclass
class SeparateReasoningReqInput(BaseReq):
    text: str  # 要解析的文本。
    reasoning_parser: str  # 指定解析器类型，例如 "deepseek-r1"。


@dataclass
class VertexGenerateReqInput(BaseReq):
    instances: List[dict]
    parameters: Optional[dict] = None


@dataclass
class RpcReqInput(BaseReq):
    method: str
    parameters: Optional[Dict] = None


@dataclass
class RpcReqOutput(BaseReq):
    success: bool
    message: str


@dataclass
class LoadLoRAAdapterReqInput(BaseReq):
    # 要新加载的 LoRA 模块名称。
    lora_name: str
    # 加载路径。
    lora_path: str
    # 是否将 LoRA 适配器固定在内存中。
    pinned: bool = False
    # LoRA 适配器的唯一标识符，在 `TokenizerManager` 中自动生成。
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path=self.lora_path,
            pinned=self.pinned,
        )


@dataclass
class UnloadLoRAAdapterReqInput(BaseReq):
    # 要卸载的 LoRA 模块名称。
    lora_name: str
    # LoRA 适配器的唯一标识符，在 `TokenizerManager` 中自动生成。
    lora_id: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
        )


@dataclass
class LoadLoRAAdapterFromTensorsReqInput(BaseReq):
    lora_name: str
    config_dict: Dict[str, Any]
    serialized_tensors: str
    pinned: bool = False
    added_tokens_config: Optional[Dict[str, Any]] = None
    lora_id: Optional[str] = None
    load_format: Optional[str] = None

    def to_ref(self) -> LoRARef:
        return LoRARef(
            lora_id=self.lora_id,
            lora_name=self.lora_name,
            lora_path="__tensor__",
            pinned=self.pinned,
        )


@dataclass
class LoRAUpdateOutput(BaseReq):
    success: bool
    error_message: Optional[str] = None
    loaded_adapters: Optional[Dict[str, LoRARef]] = None


LoadLoRAAdapterReqOutput = UnloadLoRAAdapterReqOutput = (
    LoadLoRAAdapterFromTensorsReqOutput
) = LoRAUpdateOutput


class BlockReqType(Enum):
    BLOCK = 1
    UNBLOCK = 2


@dataclass
class BlockReqInput(BaseReq):
    type: BlockReqType


@dataclass
class MemoryMetrics:
    """内存分布指标。"""

    weight_gb: float = field(
        metadata={"metric": ("gauge", "Model weight memory in GB")}
    )
    kv_cache_gb: float = field(metadata={"metric": ("gauge", "KV cache memory in GB")})
    graph_gb: float = field(metadata={"metric": ("gauge", "CUDA graph memory in GB")})
    token_capacity: int = field(
        metadata={"metric": ("gauge", "Max tokens in KV cache")}
    )


@dataclass
class SpeculativeMetrics:
    """推测解码指标。"""

    accept_length: float = field(
        metadata={"metric": ("gauge", "Avg accepted tokens per step")}
    )
    accept_rate: float = field(
        metadata={"metric": ("gauge", "Speculative acceptance rate")}
    )


@dataclass
class LoRAMetrics:
    """LoRA 适配器池指标。"""

    slots_used: int = field(metadata={"metric": ("gauge", "LoRA adapter slots in use")})
    slots_total: int = field(metadata={"metric": ("gauge", "Total LoRA adapter slots")})
    utilization: float = field(
        metadata={"metric": ("gauge", "LoRA pool utilization ratio")}
    )


@dataclass
class DisaggregationMetrics:
    """PD 分离式指标。"""

    mode: str  # "prefill"、"decode" 或 "null" - 不是指标
    prefill_prealloc_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Prefill prealloc queue requests")}
    )
    prefill_inflight_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Prefill inflight queue requests")}
    )
    decode_prealloc_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode prealloc queue requests")}
    )
    decode_transfer_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode transfer queue requests")}
    )
    decode_retracted_queue_reqs: int = field(
        default=0, metadata={"metric": ("gauge", "Decode retracted queue requests")}
    )
    kv_transfer_speed_gb_s: float = field(
        default=0.0, metadata={"metric": ("gauge", "KV transfer speed in GB/s")}
    )
    kv_transfer_latency_ms: float = field(
        default=0.0, metadata={"metric": ("gauge", "KV transfer latency in ms")}
    )


@dataclass
class QueueMetrics:
    """详细的队列分布。"""

    waiting: int = field(metadata={"metric": ("gauge", "Main waiting queue size")})
    grammar: int = field(
        metadata={"metric": ("gauge", "Grammar compilation queue size")}
    )
    paused: int = field(
        metadata={"metric": ("gauge", "Requests paused by weight sync")}
    )
    retracted: int = field(metadata={"metric": ("gauge", "Retracted requests count")})


@dataclass
class GetLoadsReqInput(BaseReq):
    """/v1/loads 端点的请求。"""

    VALID_SECTIONS = frozenset(
        {"core", "memory", "spec", "lora", "disagg", "queues", "all"}
    )

    include: List[str] = field(default_factory=lambda: ["all"])
    dp_rank: Optional[int] = None

    def __post_init__(self):
        """验证 include 部分。"""
        if self.include:
            invalid = set(self.include) - self.VALID_SECTIONS
            if invalid:
                raise ValueError(
                    f"Invalid include sections: {invalid}. "
                    f"Valid options: {sorted(self.VALID_SECTIONS)}"
                )


@dataclass
class GetLoadsReqOutput(BaseReq):
    """每个 DP rank 的负载指标，用于 /v1/loads 端点。"""

    dp_rank: int
    timestamp: float

    num_running_reqs: int = field(
        metadata={"metric": ("gauge", "Number of running requests")}
    )
    num_waiting_reqs: int = field(
        metadata={"metric": ("gauge", "Number of waiting requests")}
    )
    num_used_tokens: int = field(
        metadata={"metric": ("gauge", "Number of tokens in use")}
    )
    # num_used_tokens + 待处理的预填充 token（waiting-queue 序列长度，包括
    # 分离式 bootstrap/prealloc/transfer 队列）。用于 DP 平衡。
    num_total_tokens: int = field(
        metadata={"metric": ("gauge", "Used tokens plus pending prefill tokens")}
    )
    max_total_num_tokens: int = field(
        metadata={"metric": ("gauge", "Maximum token capacity")}
    )
    # FIXME: token_usage 实际上是所有池（KV、SWA、mamba）的最大使用量，
    # 不仅仅是 KV token 使用量。重命名需要 API 弃用流程。
    token_usage: float = field(metadata={"metric": ("gauge", "Token pool usage ratio")})
    gen_throughput: float = field(
        metadata={"metric": ("gauge", "Generation throughput tokens/sec")}
    )
    cache_hit_rate: float = field(
        metadata={"metric": ("gauge", "Prefix cache hit rate")}
    )
    utilization: float = field(
        metadata={"metric": ("gauge", "Overall utilization ratio")}
    )
    max_running_requests: int = field(
        metadata={"metric": ("gauge", "Maximum running requests capacity")}
    )

    memory: Optional[MemoryMetrics] = None
    speculative: Optional[SpeculativeMetrics] = None
    lora: Optional[LoRAMetrics] = None
    disaggregation: Optional[DisaggregationMetrics] = None
    queues: Optional[QueueMetrics] = None


@dataclass
class WatchLoadUpdateReq(BaseReq):
    loads: List[GetLoadsReqOutput]


@dataclass
class SetInjectDumpMetadataReqInput(BaseReq):
    dump_metadata: Dict[str, Any]


@dataclass
class SetInjectDumpMetadataReqOutput(BaseReq):
    success: bool


@dataclass
class LazyDumpTensorsReqInput(BaseReq):
    pass


@dataclass
class LazyDumpTensorsReqOutput(BaseReq):
    success: bool


@dataclass
class DumperControlReqInput(BaseReq):
    method: str
    body: Dict[str, Any]


@dataclass
class DumperControlReqOutput(BaseReq):
    success: bool
    response: List[Dict[str, Any]]
    error: str = ""


def _check_all_req_types():
    """辅助函数，检查所有请求类型是否在此文件中定义。"""
    import inspect
    import sys

    all_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    for class_type in all_classes:
        # 检查类名
        name = class_type[0]
        is_io_struct = (
            name.endswith("Req") or name.endswith("Input") or name.endswith("Output")
        )
        is_base_req = issubclass(class_type[1], BaseReq) or issubclass(
            class_type[1], BaseBatchReq
        )
        if is_io_struct and not is_base_req:
            raise ValueError(f"{name} is not a subclass of BaseReq or BaseBatchReq.")
        if is_base_req and not is_io_struct:
            raise ValueError(
                f"{name} is a subclass of BaseReq but not follow the naming convention."
            )


_check_all_req_types()
