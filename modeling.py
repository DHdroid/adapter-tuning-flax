from typing import Callable, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen import partitioning as nn_partitioning
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers.models.bert.configuration_bert import BertConfig
from transformers.configuration_utils import PretrainedConfig

from transformers.models.bert.modeling_flax_bert import (
    FlaxBertAttention,
    FlaxBertIntermediate,
    FlaxBertEmbeddings,
    FlaxBertPooler,
    FlaxBertOnlyMLMHead,
    FlaxBertForMaskedLMModule,
    FlaxBertPreTrainedModel,
    FlaxBertOutput,
    FlaxBertLayer,
    FlaxBertLayerCollection,
    FlaxBertEncoder,
    FlaxBertModule
)
from transformers.modeling_flax_utils import (
    FlaxPreTrainedModel,
)

remat = nn_partitioning.remat


class AdapterBertConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        num_adapters=1,
        adapter_reduce_factor=2,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.num_adapters = num_adapters
        self.adapter_reduce_factor = adapter_reduce_factor


class FlaxAdapterLayer(nn.Module):
    config: AdapterBertConfig

    def setup(self):
        self.down_proj = nn.Dense(
            self.config.hidden_size // self.config.adapter_reduce_factor,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range) # 0.02
        )
        self.act = nn.relu
        self.up_proj = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range) # 0.02
        )

    def __call__(self, hidden_states, residual):
        hidden_states = self.down_proj(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.up_proj(hidden_states)
        return hidden_states + residual


class FlaxAdapterBertOutput(FlaxBertOutput):
    config: AdapterBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.adapters = [FlaxAdapterLayer(self.config, name=f"bert_adapter_{i}") for i in range(self.config.num_adapters)]

    def __call__(self, hidden_states, attention_output, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        residual = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(residual + attention_output)

        for adapter in self.adapters:
            hidden_states = adapter(hidden_states, residual)

        hidden_states = self.LayerNorm(residual + hidden_states)
        return hidden_states


class FlaxAdapterBertLayer(FlaxBertLayer):
    config: AdapterBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxBertAttention(self.config, causal=self.config.is_decoder, dtype=self.dtype)
        self.intermediate = FlaxBertIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxAdapterBertOutput(self.config, dtype=self.dtype)
        if self.config.add_cross_attention:
            self.crossattention = FlaxBertAttention(self.config, causal=False, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        layer_head_mask,
        encoder_hidden_states: Optional[jnp.ndarray] = None,
        encoder_attention_mask: Optional[jnp.ndarray] = None,
        init_cache: bool = False,
        deterministic: bool = True,
        output_attentions: bool = False,
    ):
        # Self Attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            layer_head_mask=layer_head_mask,
            init_cache=init_cache,
            deterministic=deterministic,
            output_attentions=output_attentions,
        )
        attention_output = attention_outputs[0]

        # Cross-Attention Block
        if encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                key_value_states=encoder_hidden_states,
                deterministic=deterministic,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states, attention_output, deterministic=deterministic)


        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
            if encoder_hidden_states is not None:
                outputs += (cross_attention_outputs[1],)
        return outputs


class FlaxAdapterBertLayerCollection(FlaxBertLayerCollection):
    config: AdapterBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        if self.gradient_checkpointing:
            FlaxBertCheckpointLayer = remat(FlaxAdapterBertLayer, static_argnums=(5, 6, 7))
            self.layers = [
                FlaxBertCheckpointLayer(self.config, name=str(i), dtype=self.dtype)
                for i in range(self.config.num_hidden_layers)
            ]
        else:
            self.layers = [
                FlaxAdapterBertLayer(self.config, name=str(i), dtype=self.dtype) for i in range(self.config.num_hidden_layers)
            ]


class FlaxAdapterBertEncoder(FlaxBertEncoder):
    config: AdapterBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    gradient_checkpointing: bool = False

    def setup(self):
        self.layer = FlaxAdapterBertLayerCollection(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )


class FlaxAdapterBertModule(FlaxBertModule):
    config: AdapterBertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True
    gradient_checkpointing: bool = False

    def setup(self):
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxAdapterBertEncoder(
            self.config,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)


class FlaxAdapterBertForMaskedLMModule(FlaxBertForMaskedLMModule):
    config: AdapterBertConfig
    dtype: jnp.dtype = jnp.float32
    gradient_checkpointing: bool = False

    def setup(self):
        self.bert = FlaxAdapterBertModule(
            config=self.config,
            add_pooling_layer=False,
            dtype=self.dtype,
            gradient_checkpointing=self.gradient_checkpointing,
        )
        self.cls = FlaxBertOnlyMLMHead(config=self.config, dtype=self.dtype)


class FlaxAdapterBertForMaskedLM(FlaxBertPreTrainedModel):
    module_class = FlaxAdapterBertForMaskedLMModule


if __name__ == "__main__":
    config = AdapterBertConfig.from_pretrained("bert-base-multilingual-cased")
    config.num_adapters = 2
    config.adapter_reduce_factor = 2
    model = FlaxAdapterBertForMaskedLM.from_pretrained("bert-base-multilingual-cased", config=config)
