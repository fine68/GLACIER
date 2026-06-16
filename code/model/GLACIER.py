"""
GLACIER — a cross-modal, knowledge-enhanced LLM framework for data-efficient
cooling-load forecasting in green data centers.

Forward pipeline (see the paper / README for details):

  1. Reversible instance normalization (RevIN) of the input series.
  2. Context-Aware Temporal Synthesis (CATS) text template: domain knowledge +
     per-channel statistics and per-segment trend descriptions, rendered as text
     and embedded by the (frozen) LLM input embeddings.
  3. Text Adapter: ``k`` learnable query tokens distil the variable-length prompt
     embedding into a fixed-length text representation (a lightweight Q-Former).
  4. EGIA (Enhanced Global Interaction Attention): a cross-channel attention
     encoder over the time-series embedding, modelling inter-device coupling.
  5. KARI (Knowledge-Aligned Representation Integration): a symmetric contrastive
     loss aligning the text and time-series representation spaces.
  6. The text + series tokens are concatenated and passed through the frozen
     energy-domain LLM backbone (first ``llm_layers`` layers); a linear head with
     inverse normalization produces the forecast.

Only the lightweight adapters and the output head are trained; the LLM backbone
stays frozen. The backbone is selectable via ``configs.llm_model``
(LLAMA / GPT2 / BERT / QWEN); ``configs.llm_model_path`` points it at a local
checkpoint or a Hugging Face hub id.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import (
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
    BertConfig, BertModel, BertTokenizer,
    AutoConfig, AutoModel, AutoTokenizer,
)

from layers.Embed import DataEmbedding

transformers.logging.set_verbosity_error()


class EGIA(nn.Module):
    """Enhanced Global Interaction Attention.

    Self-attention applied over the channel dimension of a ``[B, N, D]`` tensor:
    the channel count ``N`` is used as the attention embedding size and the
    feature dimension ``D`` as the sequence length, so the block captures dynamic
    coupling *across devices/channels* rather than across time.
    """

    def __init__(self, num_channels, num_heads=1):
        super(EGIA, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=num_channels, num_heads=num_heads)

    def forward(self, x):
        # [B, N, D] -> [D, B, N]: D becomes the sequence, N the embedding dim
        x = x.transpose(1, 2).transpose(0, 1)
        attn_output, _ = self.attn(x, x, x)
        # [D, B, N] -> [B, N, D]
        return attn_output.transpose(0, 1).transpose(1, 2)


class TextAdapter(nn.Module):
    """Q-Former-style adapter: ``num_query_tokens`` learnable queries cross-attend
    over the prompt embedding and produce a fixed-length text representation.

    Returns the full set of adapted query tokens together with the first token,
    which is used as the pooled text representation for KARI alignment.
    """

    def __init__(self, hidden_size, num_query_tokens=32, num_heads=8):
        super(TextAdapter, self).__init__()
        self.num_query_tokens = num_query_tokens
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, hidden_size) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, prompt_embeddings):
        batch_size = prompt_embeddings.size(0)
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        # Q = query tokens, K/V = prompt embeddings
        q_out, _ = self.cross_attn(
            query=query_tokens, key=prompt_embeddings, value=prompt_embeddings)
        q_out = self.ln1(query_tokens + q_out)          # residual + LN
        q_out = self.ln2(q_out + self.ffn(q_out))       # FFN + residual + LN
        pooled_text = q_out[:, 0]                        # (B, hidden)
        return q_out, pooled_text


def kari_loss(align_ts, align_text, temperature=0.07):
    """KARI symmetric contrastive loss between time-series and text
    representations (positives are paired along the batch diagonal)."""
    align_ts = F.normalize(align_ts, dim=-1)
    align_text = F.normalize(align_text, dim=-1)
    logits = (align_ts @ align_text.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_ts2text = F.cross_entropy(logits, labels)
    loss_text2ts = F.cross_entropy(logits.T, labels)
    return (loss_ts2text + loss_text2ts) / 2


# Default domain-knowledge prior used in the CATS text template.
DEFAULT_DESCRIPTION = (
    "The data center cooling system includes chillers, cooling towers, heat "
    "exchangers, water pumps, and terminal air handling units, with efficiency "
    "closely tied to the wet-bulb temperature. Heat is transferred from chilled "
    "water to cooling water via chillers, dissipated through cooling towers, with "
    "lower wet-bulb temperatures enhancing efficiency. Chilled water cools and "
    "dehumidifies air via terminal units, and the cooled air is supplied to the "
    "data center. Hot air is recirculated and mixed with supply air for "
    "continuous cooling. Natural cooling techniques improve efficiency and reduce "
    "chiller load under low wet-bulb temperature conditions."
)


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in

        if self.task_name not in ('long_term_forecast', 'short_term_forecast'):
            raise NotImplementedError

        # ------------------------------------------------------------------ #
        # Frozen LLM backbone. ``configs.llm_model_path`` points at a local   #
        # checkpoint or a Hugging Face hub id; ``configs.llm_layers`` keeps    #
        # only the first N transformer layers.                                #
        # ------------------------------------------------------------------ #
        llm_path = getattr(configs, 'llm_model_path', None)

        if configs.llm_model == 'LLAMA':
            llm_path = llm_path or 'huggyllama/llama-7b'
            self.llm_config = LlamaConfig.from_pretrained(llm_path)
            self._set_llm_config(configs)
            self.llm_model = self._load(LlamaModel, llm_path)
            self.tokenizer = self._load_tokenizer(LlamaTokenizer, llm_path)
        elif configs.llm_model == 'GPT2':
            llm_path = llm_path or 'openai-community/gpt2'
            self.llm_config = GPT2Config.from_pretrained(llm_path)
            self._set_llm_config(configs)
            self.llm_model = self._load(GPT2Model, llm_path)
            self.tokenizer = self._load_tokenizer(GPT2Tokenizer, llm_path)
        elif configs.llm_model == 'BERT':
            llm_path = llm_path or 'google-bert/bert-base-uncased'
            self.llm_config = BertConfig.from_pretrained(llm_path)
            self._set_llm_config(configs)
            self.llm_model = self._load(BertModel, llm_path)
            self.tokenizer = self._load_tokenizer(BertTokenizer, llm_path)
        elif configs.llm_model == 'QWEN':
            # Energy-domain backbone used in the paper (Qwen2.5-7B architecture).
            llm_path = llm_path or 'Qwen/Qwen2.5-7B'
            self.llm_config = AutoConfig.from_pretrained(llm_path, trust_remote_code=True)
            self._set_llm_config(configs)
            self.llm_model = self._load(AutoModel, llm_path)
            self.tokenizer = self._load_tokenizer(AutoTokenizer, llm_path)
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # Freeze the LLM backbone; only adapters + head are trained.
        for param in self.llm_model.parameters():
            param.requires_grad = False

        self.description = configs.content if configs.prompt_domain else DEFAULT_DESCRIPTION

        self.hidden_size = self.llm_model.config.hidden_size

        # Trainable components.
        self.data_embedding = DataEmbedding(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.egia = EGIA(self.enc_in, num_heads=1)
        self.text_adapter = TextAdapter(self.hidden_size, num_query_tokens=32, num_heads=8)
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    # --------------------------------------------------------------------- #
    # backbone loading helpers
    # --------------------------------------------------------------------- #
    def _set_llm_config(self, configs):
        self.llm_config.num_hidden_layers = configs.llm_layers
        self.llm_config.output_attentions = True
        self.llm_config.output_hidden_states = True

    def _load(self, model_cls, path):
        """Load the backbone, trying local files first then falling back to download."""
        try:
            return model_cls.from_pretrained(
                path, trust_remote_code=True, local_files_only=True, config=self.llm_config)
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            return model_cls.from_pretrained(
                path, trust_remote_code=True, local_files_only=False, config=self.llm_config)

    @staticmethod
    def _load_tokenizer(tokenizer_cls, path):
        try:
            return tokenizer_cls.from_pretrained(
                path, trust_remote_code=True, local_files_only=True)
        except EnvironmentError:
            print("Local tokenizer files not found. Attempting to download...")
            return tokenizer_cls.from_pretrained(
                path, trust_remote_code=True, local_files_only=False)

    # --------------------------------------------------------------------- #
    # forward
    # --------------------------------------------------------------------- #
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ('long_term_forecast', 'short_term_forecast'):
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # --- (1) reversible instance normalization (RevIN) ---
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        B, T, N = x_enc.size()

        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B, T, N)

        # --- (2) build the CATS text template from statistics + segment trends ---
        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values

        diff_x_enc = x_enc.diff(dim=1)  # (B, T-1, N)
        num_segments = 4
        segment_length = x_enc.shape[1] // num_segments
        trends = []
        for b in range(x_enc.shape[0]):
            segment_trends = []
            for n in range(x_enc.shape[2]):
                feature_trends = []
                for i in range(num_segments):
                    start_idx = i * segment_length
                    end_idx = (i + 1) * segment_length if i < num_segments - 1 else x_enc.shape[1]
                    segment_diff_sum = diff_x_enc[b, start_idx:end_idx, n].sum()
                    feature_trends.append('upward' if segment_diff_sum > 0 else 'downward')
                segment_trends.append(feature_trends)
            trends.append(segment_trends)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = ', '.join([f"{val:.2f}" for val in min_values[b].tolist()])
            max_values_str = ', '.join([f"{val:.2f}" for val in max_values[b].tolist()])
            median_values_str = ', '.join([f"{val:.2f}" for val in medians[b].tolist()])
            trend_str = ', '.join(
                [f"Feature {i + 1}: {'; '.join(trends[b][i])}" for i in range(len(trends[b]))])
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {trend_str}, "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {trend_str}, "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f" trend of input is {trend_str}, "
            )
            prompt.append(prompt_)

        # restore the series to [B, T, N] then orient to [B, N, T] for embedding
        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True,
            max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))

        # --- (3) Text Adapter: distil prompt into fixed-length text tokens ---
        text_tokens, align_text = self.text_adapter(prompt_embeddings)

        # --- (4) EGIA time-series encoder ---
        x_enc = x_enc.permute(0, 2, 1).contiguous()              # [B, N, T]
        enc_out = self.data_embedding(x_enc.to(torch.bfloat16))  # [B, N, d_model]
        enc_out = self.egia(enc_out)
        align_ts = enc_out.mean(dim=1)                           # [B, d_model]

        # --- (5) KARI contrastive alignment ---
        contrastive_loss = kari_loss(align_ts, align_text, temperature=0.07)

        # --- (6) concat text + series tokens, run the frozen LLM, project ---
        llm_inputs = torch.cat([text_tokens, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llm_inputs).last_hidden_state
        dec_out = self.projection(dec_out)
        dec_out = dec_out[:, -N:, :].permute(0, 2, 1)            # keep the N series tokens
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, contrastive_loss
