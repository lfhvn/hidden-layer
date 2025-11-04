"""
RosettaModel: Multi-model wrapper for Cache-to-Cache communication

Orchestrates communication between multiple LLMs through KV-Cache projection,
enabling direct semantic transfer without text generation.

Based on the paper: "Cache-to-Cache: Direct Semantic Communication Between Large Language Models"
https://arxiv.org/abs/2510.03215
"""

import torch
import torch.nn as nn
from typing import List, Optional, Union, Dict
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from .c2c_projector import C2CProjector


class RosettaModel(nn.Module):
    """
    Wrapper for multiple LLMs that enables cache-to-cache communication.

    The model maintains:
    - A list of LLM models (base model + source models)
    - A list of C2C projectors for KV-Cache transformation
    - Configuration for layer-wise projection mappings
    - KV-Cache management across models

    Architecture:
    1. Prefill phase: Run all models in parallel to build KV-Caches
    2. Projection phase: Transform source caches to target space
    3. Fusion phase: Integrate projected caches into base model
    4. Decode phase: Generate with fused caches
    """

    def __init__(
        self,
        model_list: List[PreTrainedModel],
        base_model_idx: int = 0,
        projector_list: List[C2CProjector] = None,
    ):
        """
        Initialize RosettaModel.

        Args:
            model_list: List of PreTrainedModel instances
                        [base_model, source_model_1, source_model_2, ...]
            base_model_idx: Index of the base model (default: 0)
            projector_list: List of C2CProjector instances for cache transformation
        """
        super().__init__()

        self.base_model_idx = base_model_idx
        self.model_list = nn.ModuleList(model_list)

        device = model_list[base_model_idx].device
        dtype = model_list[base_model_idx].dtype

        if projector_list is None:
            projector_list = []
        self.projector_list = nn.ModuleList(projector_list).to(device=device, dtype=dtype)

        # Projector configuration: maps (source_model_idx, source_layer_idx, target_layer_idx) -> projector_idx
        self.projector_dict = {}
        # KV-Cache storage
        self.kv_cache_dict = {}

    @property
    def device(self):
        """Get device of base model."""
        return self.model_list[self.base_model_idx].device

    def to(self, device):
        """Move all models and projectors to device."""
        super().to(device)
        for model in self.model_list:
            model.to(device)
        for projector in self.projector_list:
            projector.to(device)
        return self

    def set_projector_config(
        self,
        source_model_idx: int,
        source_model_layer_idx: int,
        target_model_idx: int,
        target_model_layer_idx: int,
        projector_idx: int,
    ):
        """
        Configure layer-wise projection mapping.

        Args:
            source_model_idx: Index of source model
            source_model_layer_idx: Layer index in source model
            target_model_idx: Index of target model
            target_model_layer_idx: Layer index in target model
            projector_idx: Index of projector to use
        """
        if target_model_idx not in self.projector_dict:
            self.projector_dict[target_model_idx] = {}
        if source_model_idx not in self.projector_dict[target_model_idx]:
            self.projector_dict[target_model_idx][source_model_idx] = {}

        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(
            target_model_layer_idx
        )
        if layer_entry is None:
            self.projector_dict[target_model_idx][source_model_idx][
                target_model_layer_idx
            ] = [(source_model_layer_idx, projector_idx)]
        else:
            layer_entry.append((source_model_layer_idx, projector_idx))

    def get_projector(
        self,
        source_model_idx: int,
        source_model_layer_idx: int,
        target_model_idx: int,
        target_model_layer_idx: int,
    ) -> C2CProjector:
        """Get projector for specified layer mapping."""
        pair_list = self.projector_dict[target_model_idx][source_model_idx][
            target_model_layer_idx
        ]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")

        # Prefer exact source layer match
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]

        # Fallback: return first projector
        return self.projector_list[pair_list[0][1]]

    def forward(
        self,
        kv_cache_index: Optional[List[torch.Tensor]] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass with cache-to-cache communication.

        Args:
            kv_cache_index: List of tensors indicating source/target model for each token
                           Each tensor: (B, section_len, 2) where [:,:,0] = source_idx, [:,:,1] = target_idx
                           Use [-1, 0] for no projection (base model only)
            input_ids: Input token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            position_ids: Position IDs (B, seq_len)
            past_key_values: Past KV-Cache for continuation
            use_cache: Whether to return KV-Cache
            output_attentions: Whether to output attentions
            output_hidden_states: Whether to output hidden states

        Returns:
            CausalLMOutputWithPast with logits and updated caches
        """
        # Reset cache dict
        self.kv_cache_dict = {}

        _, seqlen = input_ids.size() if input_ids is not None else (0, 0)
        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1

        # Calculate section boundaries
        if kv_cache_index is not None:
            section_lengths = [kv_cache_index[i].shape[1] for i in range(num_sections)]
        else:
            section_lengths = [seqlen]

        section_starts = [0]
        for l in section_lengths:
            section_starts.append(section_starts[-1] + l)

        curr_base_kv_cache = past_key_values

        # Process each section
        if seqlen >= 1:
            for i in range(num_sections):
                start = section_starts[i]
                end = section_starts[i + 1]

                # Get section inputs
                section_input_ids = input_ids[:, start:end] if input_ids is not None else None
                section_attention_mask = (
                    attention_mask[:, :end] if attention_mask is not None else None
                )
                section_position_ids = (
                    position_ids[:, start:end] if position_ids is not None else None
                )

                # Run base model on this section
                output = self.model_list[self.base_model_idx].forward(
                    input_ids=section_input_ids,
                    attention_mask=section_attention_mask,
                    position_ids=section_position_ids,
                    past_key_values=curr_base_kv_cache,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    **kwargs,
                )

                # Update base model cache
                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                self.kv_cache_dict[self.base_model_idx][
                    self.base_model_idx
                ] = output.past_key_values
                curr_base_kv_cache = output.past_key_values

                # Run source models if not last section
                if i != num_sections - 1:
                    for source_model_idx in range(1, len(self.model_list)):
                        if self.base_model_idx not in self.kv_cache_dict:
                            self.kv_cache_dict[self.base_model_idx] = {}
                        if (
                            source_model_idx
                            not in self.kv_cache_dict[self.base_model_idx]
                        ):
                            self.kv_cache_dict[self.base_model_idx][
                                source_model_idx
                            ] = None

                        model = self.model_list[source_model_idx]
                        was_training = model.training

                        try:
                            if was_training:
                                model.eval()

                            with torch.no_grad():
                                source_output = model(
                                    input_ids=section_input_ids,
                                    attention_mask=section_attention_mask,
                                    position_ids=section_position_ids,
                                    past_key_values=self.kv_cache_dict[
                                        self.base_model_idx
                                    ][source_model_idx],
                                    use_cache=True,
                                    return_dict=True,
                                )
                                curr_source_kv_cache = source_output.past_key_values
                        finally:
                            if was_training:
                                model.train()

                        self.kv_cache_dict[self.base_model_idx][
                            source_model_idx
                        ] = curr_source_kv_cache

                # Apply projections if configured
                if (
                    self.base_model_idx in self.projector_dict
                    and kv_cache_index is not None
                ):
                    source_model_idx = kv_cache_index[i][0][0][0].item()

                    if source_model_idx != -1:
                        # Project and fuse caches
                        for (
                            target_layer_idx,
                            entry,
                        ) in self.projector_dict[self.base_model_idx][
                            source_model_idx
                        ].items():
                            base_key, base_value = curr_base_kv_cache[target_layer_idx]
                            section_base_key = base_key[:, :, start:end, :]
                            section_base_value = base_value[:, :, start:end, :]

                            for source_layer_idx, projector_idx in entry:
                                source_key, source_value = self.kv_cache_dict[
                                    self.base_model_idx
                                ][source_model_idx][source_layer_idx]
                                section_source_key = source_key[:, :, start:end, :]
                                section_source_value = source_value[:, :, start:end, :]

                                # Project source cache to target space
                                proj_key, proj_value = self.projector_list[
                                    projector_idx
                                ].forward(
                                    (section_source_key, section_source_value),
                                    (section_base_key, section_base_value),
                                )

                                # Update base cache with projected values
                                curr_base_kv_cache.key_cache[target_layer_idx][
                                    :, :, start:end, :
                                ] = proj_key
                                curr_base_kv_cache.value_cache[target_layer_idx][
                                    :, :, start:end, :
                                ] = proj_value

                        output.past_key_values = curr_base_kv_cache

        return output

    @torch.no_grad()
    def generate(
        self,
        kv_cache_index: List[torch.Tensor],
        input_ids: torch.LongTensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generate text with cache-to-cache communication.

        Args:
            kv_cache_index: List of cache index tensors
            input_ids: Input token IDs (B, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            attention_mask: Attention mask (B, seq_len)

        Returns:
            Generated token IDs (B, seq_len + max_new_tokens)
        """
        batch_size, prompt_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Prefill phase
        prefill_output = self.forward(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

        current_past = prefill_output.past_key_values
        all_input_ids = input_ids
        current_attention_mask = attention_mask

        # Get last logits for first token generation
        last_logits = prefill_output.logits[:, -1, :]

        # Generation loop
        for _ in range(max_new_tokens):
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(last_logits / temperature, dim=-1)
                if top_k > 0:
                    # Top-k sampling
                    top_k_probs, top_k_indices = torch.topk(probs, top_k)
                    probs = torch.zeros_like(probs).scatter_(1, top_k_indices, top_k_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                if top_p < 1.0:
                    # Nucleus sampling
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs > top_p
                    mask[:, 0] = False  # Keep at least one token
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)

                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy sampling
                next_token = torch.argmax(last_logits, dim=-1, keepdim=True)

            # Append token
            all_input_ids = torch.cat([all_input_ids, next_token], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((batch_size, 1), device=current_attention_mask.device),
                ],
                dim=1,
            )

            # Decode one step
            decode_kv_cache_index = [
                torch.tensor([[-1, 0]], dtype=torch.long)
                .repeat(1, 1)
                .unsqueeze(0)
                .to(input_ids.device)
            ]

            decode_output = self.forward(
                kv_cache_index=decode_kv_cache_index,
                input_ids=next_token,
                attention_mask=current_attention_mask,
                past_key_values=current_past,
                use_cache=True,
                **kwargs,
            )

            current_past = decode_output.past_key_values
            last_logits = decode_output.logits[:, -1, :]

        return all_input_ids
