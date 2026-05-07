import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from candidate_scorers import QuotientLiteCandidateScorer


CANDIDATE_FEATURE_INDEX = {
    "travel_dist_norm": 0,
    "demand_ratio": 1,
    "load_after_ratio": 2,
    "dist_to_depot_norm": 3,
    "depot_angle_diff_norm": 4,
}


def _resolve_slow_start_alpha(progress, slow_start_ratio, start_alpha, end_alpha):
    if slow_start_ratio <= 0:
        return float(end_alpha)
    clamped_progress = min(max(float(progress), 0.0), 1.0)
    ramp = min(clamped_progress / float(slow_start_ratio), 1.0)
    return float(start_alpha) + (float(end_alpha) - float(start_alpha)) * ramp


class CVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.force_first_move = model_params['force_first_move']
        self.selected_node_static_feature_names = list(model_params.get('selected_node_static_feature_names', []))
        self.selected_node_static_feature_dim = len(self.selected_node_static_feature_names)

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.encoded_nodes = None
        self.last_candidate_aux_metrics = None
        self.set_module_slow_start_progress(1.0)

    def pre_forward(self, reset_state, z):
        depot_xy = reset_state.depot_xy
        node_xy = reset_state.node_xy
        node_demand = reset_state.node_demand
        node_static = self._build_node_static_features(depot_xy, node_xy, reset_state)
        if node_static is None:
            node_xy_demand = torch.cat((node_xy, node_demand[:, :, None]), dim=2)
        else:
            node_xy_demand = torch.cat((node_xy, node_demand[:, :, None], node_static), dim=2)

        self.encoded_nodes = self.encoder(depot_xy, node_xy_demand)
        self.decoder.set_kv(self.encoded_nodes, z)

    def _build_node_static_features(self, depot_xy, node_xy, reset_state=None):
        if self.selected_node_static_feature_dim <= 0:
            return None

        features = []
        depot_dist = torch.norm(node_xy - depot_xy, dim=2)
        knn_nearest = knn_mean = knn_min = None
        if any(name.startswith("knn_") for name in self.selected_node_static_feature_names):
            dist_cust = getattr(reset_state, "customer_distance_matrix", None) if reset_state is not None else None
            if dist_cust is None:
                dist_cust = torch.cdist(node_xy, node_xy)
            dist_scale = dist_cust.amax(dim=2).amax(dim=1).clamp_min(1e-6)
            batch_size, n_nodes, _ = dist_cust.shape
            if n_nodes <= 1:
                zeros = dist_cust.new_zeros((batch_size, n_nodes))
                knn_nearest = knn_mean = knn_min = zeros
            else:
                k_eff = max(1, min(5, n_nodes - 1))
                eye = torch.eye(n_nodes, device=dist_cust.device, dtype=torch.bool).unsqueeze(0)
                dist_masked = dist_cust.masked_fill(eye, float('inf'))
                knn_vals, _ = torch.topk(dist_masked, k_eff, dim=-1, largest=False)
                knn_vals = torch.nan_to_num(knn_vals, nan=0.0, posinf=0.0, neginf=0.0)
                knn_nearest = knn_vals[..., 0] / dist_scale[:, None]
                knn_mean = knn_vals.mean(dim=-1) / dist_scale[:, None]
                knn_min = knn_vals.min(dim=-1).values / dist_scale[:, None]

        for name in self.selected_node_static_feature_names:
            if name == "knn_nearest_dist_norm":
                features.append(knn_nearest)
            elif name == "knn_mean_dist_norm":
                features.append(knn_mean)
            elif name == "knn_min_dist_norm":
                features.append(knn_min)
            elif name == "dist_to_depot_norm":
                features.append(depot_dist)
            else:
                raise ValueError(f"Unsupported node static feature: {name}")

        return torch.stack(features, dim=2)

    def forward(self, state, greedy_construction=False, EAS_incumbent_action=None):
        batch_size = state.BATCH_IDX.size(0)
        rollout_size = state.BATCH_IDX.size(1)
        problem_size = self.encoded_nodes.size(1) - 1
        self.last_candidate_aux_metrics = None
        device = state.BATCH_IDX.device

        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, rollout_size), dtype=torch.long, device=device)
            prob = torch.ones(size=(batch_size, rollout_size), device=device)

        elif state.selected_count == 1 and self.force_first_move:
            selected = torch.arange(problem_size, device=device).repeat(rollout_size // problem_size)[None, :].expand(batch_size, rollout_size)
            prob = torch.ones(size=(batch_size, rollout_size), device=device)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            probs = self.decoder(
                encoded_last_node,
                state.load,
                ninf_mask=state.ninf_mask,
                candidate_features=getattr(state, "candidate_features", None),
            )
            self.last_candidate_aux_metrics = getattr(self.decoder, "last_forward_aux", None)

            if not greedy_construction:
                while True:
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * rollout_size, -1).multinomial(1).squeeze(dim=1).reshape(batch_size, rollout_size)

                    if EAS_incumbent_action is not None:
                        selected[:, -1] = EAS_incumbent_action

                    prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)
                    if (prob != 0).all():
                        break
            else:
                selected = probs.argmax(dim=2)
                prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)

        return selected, prob

    def set_module_slow_start_progress(self, progress):
        self.encoder.set_runtime_progress(progress)
        self.decoder.set_runtime_progress(progress)

    def get_module_slow_start_state(self):
        state = dict(self.encoder.get_runtime_state())
        state.update(self.decoder.get_runtime_state())
        return state


def _get_encoding(encoded_nodes, node_index_to_pick):
    batch_size = node_index_to_pick.size(0)
    rollout_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, rollout_size, embedding_dim)
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        self.use_depth_mixer = self.model_params.get('use_depth_mixer', False)
        node_static_dim = len(self.model_params.get('selected_node_static_feature_names', []))
        self.node_static_embedding_mode = self.model_params.get('node_static_embedding_mode', 'concat')
        if self.node_static_embedding_mode not in {'concat', 'residual'}:
            raise ValueError(f"Unsupported node_static_embedding_mode: {self.node_static_embedding_mode}")
        self.node_static_slow_start_ratio = float(self.model_params.get('node_static_slow_start_ratio', 0.0))
        self.node_static_start_alpha = float(self.model_params.get('node_static_start_alpha', 1.0))
        self.runtime_node_static_alpha = 1.0

        self.embedding_depot = nn.Linear(2, embedding_dim)
        if self.node_static_embedding_mode == 'residual':
            self.embedding_node = nn.Linear(3, embedding_dim)
            self.static_residual_proj = nn.Linear(node_static_dim, embedding_dim) if node_static_dim > 0 else None
        else:
            self.embedding_node = nn.Linear(3 + node_static_dim, embedding_dim)
            self.static_residual_proj = None
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        self.depth_mixer_slow_start_ratio = float(self.model_params.get('depth_mixer_slow_start_ratio', 0.0))
        self.depth_mixer_start_alpha = float(self.model_params.get('depth_mixer_start_alpha', 0.0))
        self.runtime_depth_mixer_alpha = 1.0
        if self.use_depth_mixer:
            self.depth_key_norm = RMSNorm(embedding_dim)
            self.depth_queries = nn.Parameter(torch.empty(encoder_layer_num, embedding_dim))
            nn.init.normal_(self.depth_queries, mean=0.0, std=embedding_dim ** -0.5)
        self.set_runtime_progress(1.0)

    def _mix_history(self, history_values, history_keys, query_vector):
        depth_scores = torch.einsum('bnle,e->bnl', history_keys, query_vector)
        depth_weights = torch.softmax(depth_scores, dim=2)
        return (depth_weights.unsqueeze(-1) * history_values).sum(dim=2)

    def forward(self, depot_xy, node_xy_demand):
        embedded_depot = self.embedding_depot(depot_xy)
        if self.node_static_embedding_mode == 'residual':
            embedded_node = self.embedding_node(node_xy_demand[:, :, :3])
            if self.static_residual_proj is not None and node_xy_demand.size(2) > 3:
                static_residual = self.static_residual_proj(node_xy_demand[:, :, 3:].to(dtype=embedded_node.dtype))
                embedded_node = embedded_node + float(self.runtime_node_static_alpha) * static_residual
        else:
            embedded_node = self.embedding_node(node_xy_demand)
        out = torch.cat((embedded_depot, embedded_node), dim=1)

        if not self.use_depth_mixer:
            for layer in self.layers:
                out = layer(out)
            return out

        history_values = out.unsqueeze(2)
        history_keys = self.depth_key_norm(history_values)
        for layer_idx, layer in enumerate(self.layers):
            mixed_history = self._mix_history(history_values, history_keys, self.depth_queries[layer_idx])
            mixer_alpha = float(self.runtime_depth_mixer_alpha)
            if mixer_alpha <= 0.0:
                layer_input = history_values[:, :, -1, :]
            elif mixer_alpha >= 1.0:
                layer_input = mixed_history
            else:
                layer_input = torch.lerp(history_values[:, :, -1, :], mixed_history, mixer_alpha)
            out = layer(layer_input)
            new_value = out.unsqueeze(2)
            history_values = torch.cat((history_values, new_value), dim=2)
            history_keys = torch.cat((history_keys, self.depth_key_norm(new_value)), dim=2)
        return out

    def set_runtime_progress(self, progress):
        self.runtime_node_static_alpha = _resolve_slow_start_alpha(
            progress,
            self.node_static_slow_start_ratio,
            self.node_static_start_alpha,
            1.0,
        )
        self.runtime_depth_mixer_alpha = _resolve_slow_start_alpha(
            progress,
            self.depth_mixer_slow_start_ratio,
            self.depth_mixer_start_alpha,
            1.0,
        )
        for layer in self.layers:
            layer.set_runtime_progress(progress)

    def get_runtime_state(self):
        gate_alpha = 1.0
        if len(self.layers) > 0:
            gate_alpha = self.layers[0].runtime_gated_attention_alpha
        return {
            "node_static_alpha": float(self.runtime_node_static_alpha),
            "depth_mixer_alpha": float(self.runtime_depth_mixer_alpha),
            "gated_attention_alpha": float(gate_alpha),
        }


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.use_gated_attention = self.model_params.get('use_gated_attention', False)
        self.alpha_attn_gate = float(self.model_params.get('alpha_attn_gate', 1.0))
        self.gated_attention_scale_mode = self.model_params.get('gated_attention_scale_mode', 'sigmoid')
        self.gated_attention_slow_start_ratio = float(self.model_params.get('gated_attention_slow_start_ratio', 0.0))
        self.gated_attention_start_alpha = float(self.model_params.get('gated_attention_start_alpha', 0.0))
        self.runtime_gated_attention_alpha = self.alpha_attn_gate
        if self.use_gated_attention:
            self.W_gate = nn.Linear(embedding_dim, head_num)
            nn.init.zeros_(self.W_gate.weight)
            nn.init.constant_(self.W_gate.bias, float(self.model_params.get('gated_attention_init_bias', 2.0)))

        self.add_n_normalization_1 = AddAndInstanceNormalization(**model_params)
        self.feed_forward = FeedForward(**model_params)
        self.add_n_normalization_2 = AddAndInstanceNormalization(**model_params)

        if self.model_params['use_fast_attention']:
            self.attention_fn = fast_multi_head_attention
        else:
            self.attention_fn = multi_head_attention

    def forward_attention(self, input1):
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        headwise_out = self.attention_fn(q, k, v, return_headwise=True)
        if self.use_gated_attention:
            gate_logits = self.W_gate(input1)
            if self.gated_attention_scale_mode == 'centered_sigmoid':
                gate = 2.0 * torch.sigmoid(gate_logits)
            else:
                gate = torch.sigmoid(gate_logits)
            gate = gate.transpose(1, 2).unsqueeze(-1)
            gated_headwise_out = headwise_out * gate.to(dtype=headwise_out.dtype)
            effective_alpha = float(self.runtime_gated_attention_alpha)
            if effective_alpha >= 1.0:
                headwise_out = gated_headwise_out
            elif effective_alpha > 0.0:
                headwise_out = (1.0 - effective_alpha) * headwise_out + effective_alpha * gated_headwise_out
        out_concat = reshape_from_heads(headwise_out)
        multi_head_out = self.multi_head_combine(out_concat)
        return self.add_n_normalization_1(input1, multi_head_out)

    def forward_ffn(self, input1):
        out2 = self.feed_forward(input1)
        return self.add_n_normalization_2(input1, out2)

    def forward(self, input1):
        out1 = self.forward_attention(input1)
        return self.forward_ffn(out1)

    def set_runtime_progress(self, progress):
        self.runtime_gated_attention_alpha = _resolve_slow_start_alpha(
            progress,
            self.gated_attention_slow_start_ratio,
            self.gated_attention_start_alpha,
            self.alpha_attn_gate,
        )


class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        poly_embedding_dim = self.model_params['poly_embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        z_dim = model_params['z_dim']
        self.use_decoder_checkpointing = bool(self.model_params.get('use_decoder_checkpointing', False))

        self.use_EAS_layers = False

        self.Wq_last = nn.Linear(embedding_dim + 1, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None
        self.v = None
        self.single_head_key = None
        self.node_embeddings = None
        self.z = None

        self.use_candidate_features = bool(self.model_params.get('use_candidate_features', False))
        self.candidate_rollout_chunk_size = max(1, int(self.model_params.get('candidate_rollout_chunk_size', 32)))
        self.candidate_full_gate_slow_start_ratio = float(self.model_params.get('candidate_full_gate_slow_start_ratio', 0.0))
        self.candidate_full_gate_start_alpha = float(self.model_params.get('candidate_full_gate_start_alpha', 0.0))
        self.candidate_phi_bias_slow_start_ratio = float(self.model_params.get('candidate_phi_bias_slow_start_ratio', 0.0))
        self.candidate_phi_bias_start_alpha = float(self.model_params.get('candidate_phi_bias_start_alpha', 0.0))
        self.candidate_feature_residual_slow_start_ratio = float(self.model_params.get('candidate_feature_residual_slow_start_ratio', 0.0))
        self.candidate_feature_residual_start_alpha = float(self.model_params.get('candidate_feature_residual_start_alpha', 1.0))
        self.quotient_lite_slow_start_ratio = float(self.model_params.get('quotient_lite_slow_start_ratio', 0.0))
        self.quotient_lite_start_alpha = float(self.model_params.get('quotient_lite_start_alpha', 0.0))
        selected_candidate_feature_names = list(self.model_params.get('selected_candidate_feature_names', []))
        scorer_type = self.model_params.get('candidate_scorer_type', 'quotient_lite')
        if scorer_type not in {'baseline_additive', 'quotient_lite'}:
            raise ValueError(f"Unsupported candidate_scorer_type: {scorer_type}")
        self.candidate_scorer_type = scorer_type
        self.capture_candidate_aux = bool(self.model_params.get('capture_candidate_aux', False)) and scorer_type == 'quotient_lite'
        self.selected_candidate_feature_names = selected_candidate_feature_names
        self.selected_candidate_feature_indices = [CANDIDATE_FEATURE_INDEX[name] for name in selected_candidate_feature_names]
        self.register_buffer(
            'selected_candidate_feature_index_tensor',
            torch.tensor(self.selected_candidate_feature_indices, dtype=torch.long),
            persistent=False,
        )
        self.selected_candidate_feature_dim = len(self.selected_candidate_feature_indices)
        if self.use_candidate_features and self.selected_candidate_feature_dim <= 0:
            raise ValueError("use_candidate_features=True requires selected_candidate_feature_names")
        if self.candidate_scorer_type == 'quotient_lite' and not self.use_candidate_features:
            raise ValueError(f"candidate_scorer_type='{self.candidate_scorer_type}' requires use_candidate_features=True")

        self.poly_layer_1 = nn.Linear(embedding_dim + z_dim, poly_embedding_dim)
        self.poly_layer_2 = nn.Linear(poly_embedding_dim, embedding_dim)
        self.last_forward_aux = None

        if self.use_candidate_features:
            candidate_hidden_dim = int(self.model_params.get('candidate_feature_hidden_dim', 0))
            alpha_hidden = max(32, embedding_dim // 2)
            if candidate_hidden_dim > 0:
                self.full_phi_proj = nn.Sequential(
                    nn.Linear(self.selected_candidate_feature_dim, candidate_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(candidate_hidden_dim, embedding_dim),
                )
            else:
                self.full_phi_proj = nn.Linear(self.selected_candidate_feature_dim, embedding_dim)
            self.full_alpha_ctx_proj = nn.Linear(embedding_dim + 1, alpha_hidden, bias=False)
            self.full_alpha_node_proj = nn.Linear(embedding_dim, alpha_hidden, bias=True)
            self.full_alpha_out = nn.Linear(alpha_hidden, self.selected_candidate_feature_dim)
            self.full_phi_bias_norm = nn.LayerNorm(self.selected_candidate_feature_dim, elementwise_affine=False)
            self.full_phi_bias_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.full_gate_node = nn.Linear(embedding_dim, 1)
            self.full_gate_ctx = nn.Linear(embedding_dim + 1, 1)
            nn.init.zeros_(self.full_gate_node.weight)
            nn.init.zeros_(self.full_gate_node.bias)
            nn.init.zeros_(self.full_gate_ctx.weight)
            nn.init.constant_(self.full_gate_ctx.bias, float(self.model_params.get('candidate_full_gate_init_bias', 2.0)))
            self.full_gate_alpha = float(self.model_params.get('candidate_full_gate_alpha', 1.0))
            self.runtime_full_gate_alpha = self.full_gate_alpha
            self.runtime_phi_bias_alpha = 1.0
            self.runtime_candidate_feature_alpha = 1.0
            self.runtime_quotient_lite_alpha = 1.0
            self.quotient_lite_candidate_scorer = None
            if self.candidate_scorer_type == 'quotient_lite':
                self.quotient_lite_candidate_scorer = QuotientLiteCandidateScorer(
                    embedding_dim=embedding_dim,
                    feature_dim=self.selected_candidate_feature_dim,
                    selected_candidate_feature_names=self.selected_candidate_feature_names,
                    relative_candidate_feature_names=self.model_params.get('relative_candidate_feature_names', None),
                    zero_depot_relative_features=bool(self.model_params.get('zero_depot_relative_features', False)),
                    force_alpha_one=bool(self.model_params.get('qlite_force_alpha_one', False)),
                    disable_summary_modulation=bool(self.model_params.get('qlite_disable_summary_modulation', False)),
                    summary_dim=4,
                    hidden_dim=int(self.model_params.get('quotient_lite_hidden_dim', max(16, embedding_dim // 2))),
                    activation=self.model_params.get('quotient_scorer_activation', 'gelu'),
                )
        else:
            self.full_phi_proj = None
            self.full_alpha_ctx_proj = None
            self.full_alpha_node_proj = None
            self.full_alpha_out = None
            self.full_phi_bias_norm = None
            self.full_phi_bias_scale = None
            self.full_gate_node = None
            self.full_gate_ctx = None
            self.full_gate_alpha = 1.0
            self.runtime_full_gate_alpha = 1.0
            self.runtime_phi_bias_alpha = 1.0
            self.runtime_candidate_feature_alpha = 1.0
            self.runtime_quotient_lite_alpha = 1.0
            self.quotient_lite_candidate_scorer = None

        if self.model_params['use_fast_attention']:
            self.attention_fn = fast_multi_head_attention
        else:
            self.attention_fn = multi_head_attention

    def set_kv(self, encoded_nodes, z):
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        self.node_embeddings = encoded_nodes
        self.z = z

    def set_z(self, z):
        self.z = z

    def reset_EAS_layers(self, batch_size):
        self.EAS_W1 = torch.nn.Parameter(self.poly_layer_1.weight.mT.repeat(batch_size, 1, 1))
        self.EAS_b1 = torch.nn.Parameter(self.poly_layer_1.bias.repeat(batch_size, 1))
        self.EAS_W2 = torch.nn.Parameter(self.poly_layer_2.weight.mT.repeat(batch_size, 1, 1))
        self.EAS_b2 = torch.nn.Parameter(self.poly_layer_2.bias.repeat(batch_size, 1))
        self.use_EAS_layers = True

    def get_EAS_parameters(self):
        return [self.EAS_W1, self.EAS_b1, self.EAS_W2, self.EAS_b2]

    def _apply_shared_poly_residual(self, mh_atten_out):
        poly_out = self.poly_layer_1(torch.cat((mh_atten_out, self.z), dim=2))
        poly_out = F.relu(poly_out)
        return self.poly_layer_2(poly_out)

    def _select_candidate_features(self, candidate_features):
        if candidate_features is None or not self.use_candidate_features:
            return None
        return candidate_features.index_select(dim=-1, index=self.selected_candidate_feature_index_tensor)

    def _apply_full_candidate_reasoning(self, mh_atten_out, input_cat, selected_phi, score_static=None):
        rollout_size = mh_atten_out.size(1)
        if score_static is None:
            score_static = torch.matmul(mh_atten_out, self.single_head_key)
        gate_node = self.full_gate_node(self.node_embeddings).squeeze(-1).unsqueeze(1)
        gate_ctx_all = self.full_gate_ctx(input_cat).squeeze(-1).unsqueeze(-1)
        phi_for_bias_all = self.full_phi_bias_norm(selected_phi)
        alpha_node_proj = self.full_alpha_node_proj(self.node_embeddings).unsqueeze(1)
        alpha_ctx_proj_all = self.full_alpha_ctx_proj(input_cat)
        phi_bias_scale = self.full_phi_bias_scale.to(dtype=mh_atten_out.dtype) * float(self.runtime_phi_bias_alpha)
        phi_proj_is_linear = isinstance(self.full_phi_proj, nn.Linear)
        if phi_proj_is_linear:
            phi_proj_weight = self.full_phi_proj.weight
            phi_proj_bias = self.full_phi_proj.bias
        output_chunks = []
        for start in range(0, rollout_size, self.candidate_rollout_chunk_size):
            end = min(start + self.candidate_rollout_chunk_size, rollout_size)
            mh_chunk = mh_atten_out[:, start:end, :]
            phi_chunk = selected_phi[:, start:end, :, :]

            if phi_proj_is_linear:
                dynamic_feature_weights = torch.matmul(mh_chunk, phi_proj_weight)
                score_dynamic = (phi_chunk * dynamic_feature_weights.unsqueeze(2)).sum(dim=-1)
                if phi_proj_bias is not None:
                    score_dynamic = score_dynamic + torch.matmul(mh_chunk, phi_proj_bias).unsqueeze(-1)
            else:
                score_dynamic = torch.einsum('bre,brpe->brp', mh_chunk, self.full_phi_proj(phi_chunk))
            score = score_static[:, start:end, :] + score_dynamic

            gate_ctx = gate_ctx_all[:, start:end, :]
            gate = torch.sigmoid(gate_node + gate_ctx)
            score_gate = score * gate
            gate_alpha = float(self.runtime_full_gate_alpha)
            if gate_alpha >= 1.0:
                mixed_score = score_gate
            elif gate_alpha <= 0.0:
                mixed_score = score
            else:
                mixed_score = (1.0 - gate_alpha) * score + gate_alpha * score_gate

            phi_for_bias = phi_for_bias_all[:, start:end, :, :]
            alpha_hidden = alpha_ctx_proj_all[:, start:end, :].unsqueeze(2) + alpha_node_proj
            alpha_logits = self.full_alpha_out(F.relu(alpha_hidden))
            alpha_weights = F.softmax(alpha_logits, dim=-1)
            phi_bias = (alpha_weights * phi_for_bias).sum(dim=-1)
            output_chunks.append(mixed_score + phi_bias_scale * phi_bias)
        return torch.cat(output_chunks, dim=1)

    def _apply_quotient_lite_reasoning(self, mh_atten_out, selected_phi, candidate_features, ninf_mask, reference_score, return_aux=False):
        scorer = self.quotient_lite_candidate_scorer
        kwargs = {
            "decoder_context": mh_atten_out,
            "candidate_key": self.single_head_key.transpose(1, 2),
            "candidate_features": candidate_features,
            "ninf_mask": ninf_mask,
            "selected_phi": selected_phi,
            "return_aux": return_aux,
        }
        if reference_score is None:
            score, scorer_aux = scorer(**kwargs)
        else:
            score, scorer_aux = scorer.forward_with_reference(reference_score=reference_score, **kwargs)
        if return_aux:
            return score, scorer_aux
        return score

    def set_runtime_progress(self, progress):
        self.runtime_full_gate_alpha = _resolve_slow_start_alpha(
            progress,
            self.candidate_full_gate_slow_start_ratio,
            self.candidate_full_gate_start_alpha,
            self.full_gate_alpha,
        )
        self.runtime_phi_bias_alpha = _resolve_slow_start_alpha(
            progress,
            self.candidate_phi_bias_slow_start_ratio,
            self.candidate_phi_bias_start_alpha,
            1.0,
        )
        self.runtime_candidate_feature_alpha = _resolve_slow_start_alpha(
            progress,
            self.candidate_feature_residual_slow_start_ratio,
            self.candidate_feature_residual_start_alpha,
            1.0,
        )
        self.runtime_quotient_lite_alpha = _resolve_slow_start_alpha(
            progress,
            self.quotient_lite_slow_start_ratio,
            self.quotient_lite_start_alpha,
            1.0,
        )
        if self.quotient_lite_candidate_scorer is not None:
            self.quotient_lite_candidate_scorer.set_runtime_progress(self.runtime_quotient_lite_alpha)

    def get_runtime_state(self):
        return {
            "candidate_feature_alpha": float(self.runtime_candidate_feature_alpha),
            "candidate_gate_alpha": float(self.runtime_full_gate_alpha),
            "candidate_phi_bias_alpha": float(self.runtime_phi_bias_alpha),
            "quotient_lite_alpha": float(self.runtime_quotient_lite_alpha),
        }

    def forward(self, encoded_last_node, load, ninf_mask, candidate_features=None, return_logits=False):
        head_num = self.model_params['head_num']
        self.last_forward_aux = None
        input_cat = torch.cat((encoded_last_node, load[:, :, None]), dim=2)
        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        out_concat = self.attention_fn(q_last, self.k, self.v, rank3_ninf_mask=ninf_mask)
        mh_atten_out = self.multi_head_combine(out_concat)

        if not self.use_EAS_layers:
            poly_out = self._apply_shared_poly_residual(mh_atten_out)
        else:
            poly_out = torch.matmul(torch.cat((mh_atten_out, self.z), dim=2), self.EAS_W1)
            poly_out += self.EAS_b1[:, None]
            poly_out = F.relu(poly_out)
            poly_out = torch.matmul(poly_out, self.EAS_W2)
            poly_out += self.EAS_b2[:, None]

        mh_atten_out = mh_atten_out + poly_out

        selected_phi = None
        if candidate_features is not None and self.use_candidate_features:
            selected_phi = self._select_candidate_features(candidate_features).to(dtype=mh_atten_out.dtype)

        official_score = None
        score = None
        if selected_phi is not None:
            candidate_alpha = float(self.runtime_candidate_feature_alpha)
            qlite_alpha = float(self.runtime_quotient_lite_alpha) if self.candidate_scorer_type == 'quotient_lite' else 0.0
            need_additive = candidate_alpha > 0.0 and qlite_alpha < (1.0 - 1e-6)
            skip_reference_score = (
                self.candidate_scorer_type == 'quotient_lite'
                and qlite_alpha >= (1.0 - 1e-6)
                and not self.capture_candidate_aux
            )
            if not skip_reference_score:
                official_score = torch.matmul(mh_atten_out, self.single_head_key)
                score = official_score
            if need_additive:
                if self.use_decoder_checkpointing and self.training:
                    additive_score = checkpoint(
                        lambda a, b, c, d: self._apply_full_candidate_reasoning(a, b, c, d),
                        mh_atten_out,
                        input_cat,
                        selected_phi,
                        official_score,
                        use_reentrant=False,
                    )
                else:
                    additive_score = self._apply_full_candidate_reasoning(mh_atten_out, input_cat, selected_phi, official_score)
                if candidate_alpha >= 1.0:
                    score = additive_score
                else:
                    score = official_score + candidate_alpha * (additive_score - official_score)

            if self.candidate_scorer_type == 'quotient_lite' and qlite_alpha > 0.0:
                candidate_features_for_score = candidate_features
                ninf_mask_for_score = ninf_mask
                reference_score = None if skip_reference_score else score
                if self.use_decoder_checkpointing and self.training:
                    if skip_reference_score:
                        score = checkpoint(
                            lambda a, b, c, d: self._apply_quotient_lite_reasoning(a, b, c, d, None, return_aux=False),
                            mh_atten_out,
                            selected_phi,
                            candidate_features_for_score,
                            ninf_mask_for_score,
                            use_reentrant=False,
                        )
                    else:
                        score = checkpoint(
                            lambda a, b, c, d, e: self._apply_quotient_lite_reasoning(a, b, c, d, e, return_aux=False),
                            mh_atten_out,
                            selected_phi,
                            candidate_features_for_score,
                            ninf_mask_for_score,
                            reference_score,
                            use_reentrant=False,
                        )
                    if self.capture_candidate_aux:
                        with torch.no_grad():
                            _, scorer_aux = self._apply_quotient_lite_reasoning(
                                mh_atten_out.detach(),
                                selected_phi.detach(),
                                candidate_features_for_score.detach(),
                                ninf_mask_for_score.detach(),
                                reference_score.detach(),
                                return_aux=True,
                            )
                            scorer_aux["candidate_feature_alpha"] = torch.full(
                                (mh_atten_out.size(0), mh_atten_out.size(1), 1),
                                candidate_alpha,
                                dtype=mh_atten_out.dtype,
                                device=mh_atten_out.device,
                            )
                            self.last_forward_aux = scorer_aux
                else:
                    if not self.capture_candidate_aux:
                        score = self._apply_quotient_lite_reasoning(
                            mh_atten_out,
                            selected_phi,
                            candidate_features_for_score,
                            ninf_mask_for_score,
                            reference_score,
                            return_aux=False,
                        )
                        scorer_aux = None
                    else:
                        score, scorer_aux = self._apply_quotient_lite_reasoning(
                            mh_atten_out,
                            selected_phi,
                            candidate_features_for_score,
                            ninf_mask_for_score,
                            reference_score,
                            return_aux=True,
                        )
                    if scorer_aux is not None:
                        scorer_aux["candidate_feature_alpha"] = torch.full(
                            (mh_atten_out.size(0), mh_atten_out.size(1), 1),
                            candidate_alpha,
                            dtype=mh_atten_out.dtype,
                            device=mh_atten_out.device,
                        )
                        self.last_forward_aux = scorer_aux
        else:
            score = torch.matmul(mh_atten_out, self.single_head_key)

        score_scaled = score / self.model_params['sqrt_embedding_dim']
        score_clipped = self.model_params['logit_clipping'] * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask
        if return_logits:
            return score_masked
        return F.softmax(score_masked, dim=2)


def reshape_by_heads(qkv, head_num):
    batch_s = qkv.size(0)
    n = qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)


def reshape_from_heads(headwise_out):
    out_transposed = headwise_out.transpose(1, 2)
    batch_s = out_transposed.size(0)
    n = out_transposed.size(1)
    head_num = out_transposed.size(2)
    key_dim = out_transposed.size(3)
    return out_transposed.reshape(batch_s, n, head_num * key_dim)


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None, return_headwise=False):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    score_scaled = score / math.sqrt(key_dim)
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    out = torch.matmul(weights, v)

    if return_headwise:
        return out
    return reshape_from_heads(out)


def fast_multi_head_attention(q, k, v, rank3_ninf_mask=None, return_headwise=False):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    input_s = k.size(2)

    mask = None
    if rank3_ninf_mask is not None:
        mask = rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if return_headwise:
        return out
    return reshape_from_heads(out)


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.norm = nn.InstanceNorm1d(model_params['embedding_dim'], affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2
        normalized = self.norm(added.transpose(1, 2))
        return normalized.transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
