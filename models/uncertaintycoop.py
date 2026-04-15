import torch
import torch.nn as nn

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from copy import deepcopy
import torch.nn.functional as F

_tokenizer = _Tokenizer()

__all__ = ['uncertaintycoop', 'UncertaintyCoop']


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    model = clip.build_model_conv_proj(state_dict or model.state_dict(), cfg)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class MLCPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, prompt_type="positive"):
        """
        prompt_type: 'positive' | 'negative' | 'uncertainty'
        """
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        # 1) Select configuration parameters based on type
        if prompt_type == "positive":
            n_ctx = cfg.TRAINER.COOP_MLC.N_CTX_POS
            ctx_init = getattr(cfg.TRAINER.COOP_MLC, "POSITIVE_PROMPT_INIT", "").strip()
        elif prompt_type == "negative":
            n_ctx = getattr(cfg.TRAINER.COOP_MLC, "N_CTX_NEG", cfg.TRAINER.COOP_MLC.N_CTX_POS)
            ctx_init = getattr(cfg.TRAINER.COOP_MLC, "NEGATIVE_PROMPT_INIT", "").strip()
        elif prompt_type == "uncertainty":
            n_ctx = getattr(cfg.TRAINER.COOP_MLC, "N_CTX_UNC", cfg.TRAINER.COOP_MLC.N_CTX_POS)
            ctx_init = getattr(cfg.TRAINER.COOP_MLC, "UNCERTAINTY_PROMPT_INIT", "").strip()
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

        self.prompt_type = prompt_type
        self.n_ctx = n_ctx

        # 2) Initialize context vectors
        if ctx_init:
            # Initialize from text template
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt_tokens = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt_tokens).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
            if cfg.TRAINER.COOP_MLC.CSC:
                ctx_vectors = torch.stack([deepcopy(ctx_vectors) for _ in range(n_cls)], dim=0)
        else:
            # Random initialization
            if cfg.TRAINER.COOP_MLC.CSC:
                print(f"[{prompt_type}] Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print(f"[{prompt_type}] Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'[{prompt_type}] Initial context: "{prompt_prefix}"')
        print(f"[{prompt_type}] Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        # 3) Generate complete templates
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [f"{prompt_prefix} {name}." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding_all = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding_all[:, :1, :])
        self.register_buffer("token_suffix", embedding_all[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.register_buffer("tokenized_prompts", tokenized_prompts)
        self.name_lens = name_lens

    def forward(self, cls_id=None):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1) if cls_id is None else ctx.unsqueeze(0).expand(len(cls_id), -1, -1)
        else:
            if cls_id is not None:
                ctx = ctx[cls_id]

        prefix = self.token_prefix if cls_id is None else self.token_prefix[cls_id]
        suffix = self.token_suffix if cls_id is None else self.token_suffix[cls_id]

        prompts = torch.cat([prefix, ctx, suffix], dim=1)

        tokenized_prompts = self.tokenized_prompts if cls_id is None else self.tokenized_prompts[cls_id]
        return prompts, tokenized_prompts


class UncertaintyCoop(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.visual_encoder_type = cfg.MODEL.BACKBONE.NAME

        # Positive Prompt Learner
        self.prompt_learner_pos = MLCPromptLearner(cfg, classnames, clip_model, prompt_type="positive")
        # Uncertainty Prompt Learner
        self.prompt_learner_unc = MLCPromptLearner(cfg, classnames, clip_model, prompt_type="uncertainty")
        # Compatibility with old training code
        self.prompt_learner = self.prompt_learner_pos

        self.classnames = classnames
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = cfg.TRAINER.COOP_MLC.LS
        self.dtype = clip_model.dtype
        self.cfg = cfg
        self.clip_model = clip_model

        # Learnable vectors
        self.txt_prompt_neg = nn.Parameter(torch.randn([len(self.classnames), 512]))
        self.txt_prompt_learn = nn.Parameter(torch.randn([len(self.classnames), 512]))

        # Fusion weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))

    # ========================================
    def forward(self, image, cls_id=None, uncertainty_weight=None):
        """
        Forward with:
          - positive & uncertainty prompts both from prompt learners
          - negative prompts from learnable vectors
          - fusion formula combining all
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) Extract image features
        image_features, attn_weights = self.image_encoder(image.type(self.dtype))  # (B, D, L)

        # 2) Positive prompts
        prompts_pos, tokenized_prompts_pos = self.prompt_learner(cls_id)
        text_features_pos = self.text_encoder(prompts_pos, tokenized_prompts_pos)
        text_features_pos = text_features_pos + self.txt_prompt_learn  # Add offset

        # 3) Uncertainty prompts
        prompts_unc, tokenized_prompts_unc = self.prompt_learner_unc(cls_id)
        text_features_uncertain = self.text_encoder(prompts_unc, tokenized_prompts_unc)

        # 4) Negative prompts
        text_features_neg = self.txt_prompt_neg

        # 5) Normalize
        image_features_norm = image_features / (image_features.norm(dim=1, keepdim=True) + 1e-12)
        text_pos = text_features_pos / (text_features_pos.norm(dim=-1, keepdim=True) + 1e-12)
        text_neg = text_features_neg / (text_features_neg.norm(dim=-1, keepdim=True) + 1e-12)
        text_unc = text_features_uncertain / (text_features_uncertain.norm(dim=-1, keepdim=True) + 1e-12)

        # 6) Compute Positive + Negative logits
        text_posneg = torch.cat([text_neg, text_pos], dim=0)  # (2*n_cls, D)
        weight_posneg = text_posneg[:, :, None]  # (2*n_cls, D, 1)
        out_posneg = 20.0 * F.conv1d(image_features_norm, weight_posneg)  # (B, 2*n_cls, L)

        b, c, L = out_posneg.shape
        out_pos_half = out_posneg[:, c // 2:, :]  # (B, n_cls, L)
        w_half = F.softmax(out_pos_half, dim=-1)  # (B, n_cls, L)
        w = torch.cat([w_half, w_half], dim=1)  # (B, 2*n_cls, L)
        out_agg = 5.0 * (out_posneg * w).sum(-1)  # (B, 2*n_cls)
        logits_posneg = out_agg.reshape(b, 2, c // 2)  # (B, 2, n_cls)
        s_neg = logits_posneg[:, 0, :]  # (B, n_cls)
        s_pos = logits_posneg[:, 1, :]  # (B, n_cls)

        # 7) Compute Uncertainty logits
        weight_unc = text_unc[:, :, None]  # (n_cls, D, 1)
        out_unc = 20.0 * F.conv1d(image_features_norm, weight_unc)  # (B, n_cls, L)
        w_unc = F.softmax(out_unc, dim=-1)  # (B, n_cls, L)
        out_unc_agg = 5.0 * (out_unc * w_unc).sum(-1)  # (B, n_cls)
        s_uncertain = out_unc_agg  # (B, n_cls)

        # 8) Handle uncertainty weights
        if uncertainty_weight is None:
            uncertainty_weight = torch.zeros_like(s_pos).to(device)
        else:
            if uncertainty_weight.dim() == 3:
                uncertainty_weight = uncertainty_weight.squeeze(1)

        # 9) Fusion weights
        alpha = torch.clamp(self.alpha, 0.0, 1.0)
        beta = torch.clamp(self.beta, 0.0, 1.0)
        gamma = torch.clamp(self.gamma, 0.0, 1.0)

        # 10) Fuse final scores
        p = (1.0 - uncertainty_weight) * (alpha * s_pos + beta * (1.0 - s_neg)) \
            + uncertainty_weight * (gamma * s_uncertain)

        zeros = torch.zeros_like(p)
        final_logits = torch.stack([zeros, p], dim=1)  # (B, 2, n_cls)

        return final_logits

    # ========================================
    @property
    def network_name(self):
        return f'UncertaintyCoop-{self.visual_encoder_type}'

    def backbone_params(self):
        params = []
        for name, param in self.named_parameters():
            if "image_encoder" in name and "prompt_learner" not in name and 'attnpool' not in name:
                params.append(param)
        return params

    def attn_params(self):
        params = []
        for name, param in self.named_parameters():
            if 'attnpool' in name and 'image_encoder' in name:
                params.append(param)
                print(name)
        return params

    def prompt_params(self):
        params = []
        for name, param in self.named_parameters():
            if "prompt_learner" in name:
                params.append(param)
        return params

    def txt_new_prompt(self):
        params = []
        for name, param in self.named_parameters():
            if "txt_prompt" in name:
                params.append(param)
        return params

    def fusion_weights_params(self):
        params = []
        for name, param in self.named_parameters():
            if name in ["alpha", "beta", "gamma"]:
                params.append(param)
        return params


def uncertaintycoop(cfg, classnames, **kwargs):
    print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
    clip_model = load_clip_to_cpu(cfg)

    clip_model.float()

    print("Building positivecoop")
    model = UncertaintyCoop(cfg, classnames, clip_model)

    if not cfg.TRAINER.FINETUNE_BACKBONE:
        print('Freeze the backbone weights')
        backbone_params = model.backbone_params()
        for param in backbone_params:
            param.requires_grad_(False)

    if not cfg.TRAINER.FINETUNE_ATTN:
        print('Freeze the attn weights')
        attn_params = model.attn_params()
        for param in attn_params:
            param.requires_grad_(False)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Note that multi-gpu training could be slow because CLIP's size is
    # big, which slows down the copy operation in DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        model = nn.DataParallel(model)
    return model

