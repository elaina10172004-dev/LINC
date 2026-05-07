import os

import torch
import torch.nn as nn


_ENABLE_MASK_ASSERTS = os.environ.get("LINC_DEBUG_MASK_ASSERTS", "0") == "1"


CANDIDATE_FEATURE_INDEX = {
    "travel_dist_norm": 0,
    "dist_to_start_norm": 1,
    "dist_to_centroid_norm": 2,
    "centroid_angle_diff_norm": 3,
}


def _make_mlp(input_dim, hidden_dim, output_dim, activation="gelu"):
    if activation == "relu":
        act = nn.ReLU()
    elif activation == "gelu":
        act = nn.GELU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        act,
        nn.Linear(hidden_dim, output_dim),
    )


def _make_activation(activation="gelu"):
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {activation}")


def _masked_mean_dim(values, mask, dim):
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    return (values * mask_f).sum(dim=dim) / denom


def _masked_min_dim(values, mask, dim):
    filled = values.masked_fill(~mask, float("inf"))
    out = filled.min(dim=dim).values
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def _masked_std_dim(values, mask, dim):
    mask_f = mask.to(dtype=values.dtype)
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    mean = (values * mask_f).sum(dim=dim) / denom
    centered = values - mean.unsqueeze(dim)
    var = (centered.pow(2) * mask_f).sum(dim=dim) / denom
    return torch.sqrt(var + 1e-6)


def _selected_feature_mean_and_count(selected_phi, feasible_customer_mask):
    feature_mask = feasible_customer_mask.unsqueeze(-1).to(dtype=selected_phi.dtype)
    feasible_count = feature_mask.sum(dim=2).clamp_min(1.0)
    selected_mean = (selected_phi * feature_mask).sum(dim=2) / feasible_count
    return selected_mean, feasible_count.squeeze(-1)


def _assert_valid_ninf_mask(ninf_mask):
    if not _ENABLE_MASK_ASSERTS:
        return
    valid_mask_values = torch.logical_or(ninf_mask == 0, ~torch.isfinite(ninf_mask))
    if not bool(valid_mask_values.all().item()):
        raise AssertionError("ninf_mask must contain only 0 or non-finite values")


def _build_feasible_customer_mask(ninf_mask):
    return torch.isfinite(ninf_mask)


def _lookup_candidate_feature(selected_phi, candidate_features, feature_name, local_index_map):
    local_idx = local_index_map.get(feature_name)
    if local_idx is not None:
        return selected_phi[..., local_idx]
    global_idx = CANDIDATE_FEATURE_INDEX.get(feature_name)
    if candidate_features is not None and global_idx is not None and candidate_features.size(-1) > global_idx:
        return candidate_features[..., global_idx].to(dtype=selected_phi.dtype)
    return selected_phi.new_zeros(selected_phi.shape[:3])


def _lookup_candidate_feature_mean(selected_phi, candidate_features, feature_name, local_index_map, feasible_customer_mask, selected_mean=None):
    local_idx = local_index_map.get(feature_name)
    if local_idx is not None and selected_mean is not None:
        return selected_mean[..., local_idx]
    feature = _lookup_candidate_feature(selected_phi, candidate_features, feature_name, local_index_map)
    return _masked_mean_dim(feature, feasible_customer_mask, dim=2)


def _build_tiny_residual_summary(
    selected_phi,
    feasible_customer_mask,
    local_index_map,
    candidate_features=None,
    selected_mean=None,
    feasible_count=None,
):
    customer_count = max(selected_phi.size(2), 1)
    if feasible_count is None:
        feasible_count = feasible_customer_mask.to(dtype=selected_phi.dtype).sum(dim=2).clamp_min(1.0)
    feasible_ratio = feasible_count / float(customer_count)
    summary = torch.stack(
        [
            feasible_ratio,
            _lookup_candidate_feature_mean(
                selected_phi, candidate_features, "travel_dist_norm", local_index_map, feasible_customer_mask, selected_mean
            ),
            _lookup_candidate_feature_mean(
                selected_phi, candidate_features, "dist_to_start_norm", local_index_map, feasible_customer_mask, selected_mean
            ),
            _lookup_candidate_feature_mean(
                selected_phi, candidate_features, "dist_to_centroid_norm", local_index_map, feasible_customer_mask, selected_mean
            ),
        ],
        dim=-1,
    )
    return summary, feasible_ratio


def _collect_feature_stats(selected_phi, phi_q, feasible_customer_mask):
    feature_mask = feasible_customer_mask.unsqueeze(-1)
    return {
        "raw_feature_mean": _masked_mean_dim(selected_phi, feature_mask, dim=2),
        "raw_feature_std": _masked_std_dim(selected_phi, feature_mask, dim=2),
        "phi_q_feature_mean": _masked_mean_dim(phi_q, feature_mask, dim=2),
        "phi_q_feature_std": _masked_std_dim(phi_q, feature_mask, dim=2),
    }


def _build_relative_features(
    selected_phi,
    feasible_customer_mask,
    relative_index,
    zero_depot_relative_features=False,
):
    if relative_index.numel() == 0:
        return selected_phi

    rel_index = relative_index.to(device=selected_phi.device)
    rel_values = selected_phi.index_select(dim=-1, index=rel_index)
    rel_mask = feasible_customer_mask.unsqueeze(-1)
    rel_mean = _masked_mean_dim(rel_values, rel_mask, dim=2).unsqueeze(2).to(dtype=selected_phi.dtype)
    phi_q = selected_phi.clone()
    phi_q[..., rel_index] = rel_values - rel_mean
    if zero_depot_relative_features:
        phi_q[..., 0, rel_index] = 0
    return phi_q


def _linear_relative_dynamic_score(
    selected_phi,
    selected_mean,
    h_tilde,
    phi_proj,
    relative_index,
    zero_depot_relative_features=False,
):
    dynamic_feature_weights = torch.matmul(h_tilde, phi_proj.weight)
    dynamic_score = (selected_phi * dynamic_feature_weights.unsqueeze(2)).sum(dim=-1)
    if relative_index.numel() > 0:
        rel_index = relative_index.to(device=selected_phi.device)
        rel_mean = selected_mean.index_select(dim=-1, index=rel_index)
        rel_weights = dynamic_feature_weights.index_select(dim=-1, index=rel_index)
        rel_correction = (rel_mean * rel_weights).sum(dim=-1).unsqueeze(-1)
        dynamic_score = dynamic_score - rel_correction
        if zero_depot_relative_features:
            depot_rel_values = selected_phi[:, :, 0, :].index_select(dim=-1, index=rel_index)
            depot_rel_contrib = ((depot_rel_values - rel_mean) * rel_weights).sum(dim=-1)
            depot_correction = torch.zeros_like(dynamic_score)
            depot_correction[..., 0] = depot_rel_contrib
            dynamic_score = dynamic_score - depot_correction
    if phi_proj.bias is not None:
        dynamic_score = dynamic_score + torch.matmul(h_tilde, phi_proj.bias).unsqueeze(-1)
    return dynamic_score


class QuotientLiteCandidateScorer(nn.Module):
    DEFAULT_RELATIVE_CANDIDATE_FEATURES = [
        "travel_dist_norm",
        "dist_to_start_norm",
        "dist_to_centroid_norm",
    ]

    def __init__(
        self,
        embedding_dim,
        feature_dim,
        selected_candidate_feature_names,
        relative_candidate_feature_names=None,
        zero_depot_relative_features=False,
        force_alpha_one=False,
        disable_summary_modulation=False,
        summary_dim=4,
        hidden_dim=None,
        activation="gelu",
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.feature_dim = int(feature_dim)
        self.summary_dim = int(summary_dim)
        self.zero_depot_relative_features = bool(zero_depot_relative_features)
        self.force_alpha_one = bool(force_alpha_one)
        self.disable_summary_modulation = bool(disable_summary_modulation)
        self.selected_candidate_feature_names = list(selected_candidate_feature_names)
        self.selected_feature_name_to_local_idx = {
            name: idx for idx, name in enumerate(self.selected_candidate_feature_names)
        }
        selected_feature_indices = [CANDIDATE_FEATURE_INDEX[name] for name in self.selected_candidate_feature_names]
        self.register_buffer(
            "selected_feature_index",
            torch.tensor(selected_feature_indices, dtype=torch.long),
            persistent=False,
        )

        requested_relative = list(relative_candidate_feature_names or self.DEFAULT_RELATIVE_CANDIDATE_FEATURES)
        relative_names = [name for name in requested_relative if name in self.selected_candidate_feature_names]
        relative_positions = [self.selected_candidate_feature_names.index(name) for name in relative_names]
        self.register_buffer(
            "relative_feature_index",
            torch.tensor(relative_positions, dtype=torch.long),
            persistent=False,
        )

        hidden_dim = int(hidden_dim or max(16, self.embedding_dim // 2))
        summary_input_dim = self.embedding_dim + self.summary_dim
        self.full_phi_proj = nn.Linear(self.feature_dim, self.embedding_dim)
        self.summary_input_proj = nn.Linear(summary_input_dim, 3 * hidden_dim)
        self.summary_activation = _make_activation(activation)
        self.summary_to_gamma = nn.Linear(hidden_dim, self.embedding_dim)
        self.summary_to_beta = nn.Linear(hidden_dim, self.embedding_dim)
        self.summary_to_alpha = nn.Linear(hidden_dim, 1)
        self.runtime_quotient_alpha = 1.0

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        old_first = [prefix + f"summary_to_{name}.0" for name in ("gamma", "beta", "alpha")]
        if prefix + "summary_input_proj.weight" not in state_dict and all(k + ".weight" in state_dict for k in old_first):
            state_dict[prefix + "summary_input_proj.weight"] = torch.cat(
                [state_dict[k + ".weight"] for k in old_first],
                dim=0,
            )
            state_dict[prefix + "summary_input_proj.bias"] = torch.cat(
                [state_dict[k + ".bias"] for k in old_first],
                dim=0,
            )

        for name in ("gamma", "beta", "alpha"):
            old_prefix = prefix + f"summary_to_{name}.2"
            new_prefix = prefix + f"summary_to_{name}"
            if new_prefix + ".weight" not in state_dict and old_prefix + ".weight" in state_dict:
                state_dict[new_prefix + ".weight"] = state_dict[old_prefix + ".weight"]
            if new_prefix + ".bias" not in state_dict and old_prefix + ".bias" in state_dict:
                state_dict[new_prefix + ".bias"] = state_dict[old_prefix + ".bias"]

        for name in ("gamma", "beta", "alpha"):
            for layer_idx in (0, 2):
                for suffix in ("weight", "bias"):
                    state_dict.pop(prefix + f"summary_to_{name}.{layer_idx}.{suffix}", None)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _select_candidate_features(self, candidate_features):
        if candidate_features.size(-1) == self.feature_dim:
            return candidate_features
        return candidate_features.index_select(dim=-1, index=self.selected_feature_index)

    def set_runtime_progress(self, progress):
        self.runtime_quotient_alpha = min(max(float(progress), 0.0), 1.0)

    def get_runtime_state(self):
        return {
            "quotient_lite_alpha": float(self.runtime_quotient_alpha),
        }

    def _compute_score(
        self,
        decoder_context,
        candidate_key,
        candidate_features,
        ninf_mask,
        selected_phi=None,
        apply_runtime_alpha=True,
        return_aux=True,
    ):
        _assert_valid_ninf_mask(ninf_mask)
        feasible_customer_mask = _build_feasible_customer_mask(ninf_mask)
        if selected_phi is None:
            selected_phi = self._select_candidate_features(candidate_features)
        selected_phi = selected_phi.to(dtype=decoder_context.dtype)
        selected_mean, feasible_count = _selected_feature_mean_and_count(selected_phi, feasible_customer_mask)
        phi_q = None
        if return_aux:
            phi_q = _build_relative_features(
                selected_phi,
                feasible_customer_mask,
                self.relative_feature_index,
                zero_depot_relative_features=self.zero_depot_relative_features,
            )
        summary, feasible_ratio = _build_tiny_residual_summary(
            selected_phi,
            feasible_customer_mask,
            self.selected_feature_name_to_local_idx,
            candidate_features=candidate_features,
            selected_mean=selected_mean,
            feasible_count=feasible_count,
        )
        summary = summary.to(dtype=decoder_context.dtype)
        summary_input = torch.cat([decoder_context, summary], dim=-1)
        summary_hidden = self.summary_activation(self.summary_input_proj(summary_input))
        gamma_hidden, beta_hidden, alpha_hidden = summary_hidden.chunk(3, dim=-1)
        gamma = torch.tanh(self.summary_to_gamma(gamma_hidden))
        beta = self.summary_to_beta(beta_hidden)
        alpha = torch.sigmoid(self.summary_to_alpha(alpha_hidden))
        runtime_alpha = float(self.runtime_quotient_alpha) if apply_runtime_alpha else 1.0
        if self.disable_summary_modulation:
            gamma = torch.zeros_like(gamma)
            beta = torch.zeros_like(beta)
        else:
            gamma = gamma * runtime_alpha
            beta = beta * runtime_alpha
        if self.force_alpha_one:
            alpha = torch.ones_like(alpha)
        else:
            alpha = torch.ones_like(alpha) + runtime_alpha * (alpha - torch.ones_like(alpha))
        h_tilde = decoder_context * (1.0 + gamma) + beta

        base_score = alpha * torch.einsum("bre,bpe->brp", decoder_context, candidate_key)
        if phi_q is None:
            dynamic_score = _linear_relative_dynamic_score(
                selected_phi,
                selected_mean,
                h_tilde,
                self.full_phi_proj,
                self.relative_feature_index,
                zero_depot_relative_features=self.zero_depot_relative_features,
            )
        else:
            dynamic_feature_weights = torch.matmul(h_tilde, self.full_phi_proj.weight)
            dynamic_score = (phi_q * dynamic_feature_weights.unsqueeze(2)).sum(dim=-1)
            if self.full_phi_proj.bias is not None:
                dynamic_score = dynamic_score + torch.matmul(h_tilde, self.full_phi_proj.bias).unsqueeze(-1)
        dynamic_score = dynamic_score * runtime_alpha
        raw_score = base_score + dynamic_score
        if not return_aux:
            return raw_score, None
        feature_stats = _collect_feature_stats(selected_phi, phi_q, feasible_customer_mask)
        return raw_score, {
            "base_score": base_score,
            "dynamic_score": dynamic_score,
            "summary": summary,
            "alpha": alpha,
            "quotient_lite_alpha": torch.full(
                (decoder_context.size(0), decoder_context.size(1), 1),
                runtime_alpha,
                dtype=decoder_context.dtype,
                device=decoder_context.device,
            ),
            "feasible_ratio": feasible_ratio.to(dtype=decoder_context.dtype),
            **feature_stats,
        }

    def forward(self, decoder_context, candidate_key, candidate_features, ninf_mask, selected_phi=None, return_aux=True):
        return self._compute_score(
            decoder_context=decoder_context,
            candidate_key=candidate_key,
            candidate_features=candidate_features,
            ninf_mask=ninf_mask,
            selected_phi=selected_phi,
            apply_runtime_alpha=True,
            return_aux=return_aux,
        )

    def forward_with_reference(
        self,
        decoder_context,
        candidate_key,
        candidate_features,
        ninf_mask,
        reference_score,
        selected_phi=None,
        return_aux=True,
    ):
        raw_score, aux = self._compute_score(
            decoder_context=decoder_context,
            candidate_key=candidate_key,
            candidate_features=candidate_features,
            ninf_mask=ninf_mask,
            selected_phi=selected_phi,
            apply_runtime_alpha=False,
            return_aux=return_aux,
        )
        runtime_alpha = float(self.runtime_quotient_alpha)
        score = reference_score + runtime_alpha * (raw_score - reference_score)
        if not return_aux:
            return score, None
        aux["quotient_lite_alpha"] = torch.full(
            (decoder_context.size(0), decoder_context.size(1), 1),
            runtime_alpha,
            dtype=decoder_context.dtype,
            device=decoder_context.device,
        )
        aux["reference_score"] = reference_score
        aux["raw_quotient_score"] = raw_score
        aux["morphed_score"] = score
        return score, aux
