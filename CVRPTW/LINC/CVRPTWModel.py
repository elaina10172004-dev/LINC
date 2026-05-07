import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from candidate_scorers import QuotientLiteCandidateScorer

CANDIDATE_FEATURE_INDEX = {
    "travel_dist_norm": 0,
    "wait_norm": 1,
    "tw_slack_ratio": 2,
    "arrival_time_norm": 3,
    "departure_time_norm": 4,
    "depot_angle_diff_norm": 5,
}


def _accurate_cdist(x, y):
    if not x.is_cuda:
        return torch.cdist(x, y)

    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        return torch.cdist(x, y)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32


def _make_activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")

class CVRPTWModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.force_first_move = model_params['force_first_move']
        self.selected_node_static_feature_names = list(model_params.get('selected_node_static_feature_names', []))
        self.selected_node_static_feature_dim = len(self.selected_node_static_feature_names)

        self.encoder = CVRP_Encoder(**model_params)
        self.decoder = CVRP_Decoder(**model_params)
        self.use_learned_corrector = bool(self.model_params.get('use_learned_corrector', False))
        self.corrector_mode = self.model_params.get('corrector_mode', 'step')
        self.corrector_step_feature_dim = int(self.model_params.get('corrector_step_feature_dim', len(CANDIDATE_FEATURE_INDEX) + 5))
        if self.use_learned_corrector:
            if self.corrector_mode == 'macro_route_cut':
                self.corrector = CVRPTWMacroCorrector(**model_params)
            else:
                self.corrector = CVRPTWCorrector(**model_params)
        else:
            self.corrector = None
        self.encoded_nodes = None
        self.last_candidate_aux_metrics = None
        # shape: (batch, problem+1, EMBEDDING_DIM)

    def pre_forward(self, reset_state, z=None):
        depot_xy = reset_state.depot_xy
        # shape: (batch, 1, 2)
        node_xy = reset_state.node_xy
        # shape: (batch, problem, 2)
        node_demand = reset_state.node_demand
        # shape: (batch, problem)
        node_tw = reset_state.node_tw
        # shape: (batch, problem, 2)
        service_t = getattr(reset_state, "service_t", None)
        node_static = self._build_node_static_features(depot_xy, node_xy, node_tw)
        node_parts = [node_xy, node_demand[:, :, None], node_tw]
        if bool(self.model_params.get("include_service_duration_in_node_embedding", False)):
            if service_t is None:
                raise RuntimeError("reset_state.service_t is required when include_service_duration_in_node_embedding=True")
            if service_t.ndim == 1:
                service_t = service_t[:, None]
            service_feature = service_t[:, None, :].expand(-1, node_xy.size(1), -1)
            node_parts.append(service_feature)
        if node_static is not None:
            node_parts.append(node_static)
        node_data = torch.cat(node_parts, dim=2)
        # shape: (batch, problem, 5)


        self.encoded_nodes = self.encoder(depot_xy, node_data)
        # shape: (batch, problem+1, embedding)
        self.decoder.set_kv(self.encoded_nodes, z)

    def _build_node_static_features(self, depot_xy, node_xy, node_tw):
        if self.selected_node_static_feature_dim <= 0:
            return None

        features = []
        depot_dist = torch.norm(node_xy - depot_xy, dim=2)
        tw_length = node_tw[:, :, 1] - node_tw[:, :, 0]

        knn_nearest = knn_mean = knn_min = None
        if any(name.startswith("knn_") for name in self.selected_node_static_feature_names):
            dist_cust = _accurate_cdist(node_xy, node_xy)
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
            elif name == "tw_length_norm":
                features.append(tw_length)
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


        if state.selected_count == 0:
            selected = torch.zeros(size=(batch_size, rollout_size), dtype=torch.long)
            prob = torch.ones(size=(batch_size, rollout_size))

        elif state.selected_count == 1 and self.force_first_move:
            selected = torch.arange(problem_size).repeat(rollout_size // problem_size)[None, :].expand(batch_size,
                                                                                                    rollout_size)
            prob = torch.ones(size=(batch_size, rollout_size))

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, rollout, embedding)
            probs = self.decoder(
                encoded_last_node,
                state.load,
                state.time,
                ninf_mask=state.ninf_mask,
                candidate_features=getattr(state, "candidate_features", None),
                current_node=getattr(state, "current_node", None),
                visited_mask=getattr(state, "visited_mask", None),
                finished=getattr(state, "finished", None),
            )
            # shape: (batch, rollout, problem+1)
            self.last_candidate_aux_metrics = getattr(self.decoder, "last_forward_aux", None)

            if not greedy_construction:
                while True:  # to fix pytorch.multinomial bug on selecting 0 probability elements
                    with torch.no_grad():
                        selected = probs.reshape(batch_size * rollout_size, -1).multinomial(1) \
                            .squeeze(dim=1).reshape(batch_size, rollout_size)
                    # shape: (batch, rollout)

                    if EAS_incumbent_action is not None:
                        selected[:, -1] = EAS_incumbent_action

                    prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)
                    # shape: (batch, rollout)
                    if (prob != 0).all():
                        break


            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, rollout)
                prob = probs[state.BATCH_IDX, state.ROLLOUT_IDX, selected].reshape(batch_size, rollout_size)

        return selected, prob

    def score_correction(self, history_actions, history_step_features, history_valid_mask, removable_mask):
        if not self.use_learned_corrector or self.corrector is None:
            raise RuntimeError("Learned corrector is disabled for this model.")

        if self.encoded_nodes is None:
            raise RuntimeError("encoded_nodes are not initialized. Call pre_forward first.")

        batch_size, rollout_size, seq_len = history_actions.shape
        embedding_dim = self.encoded_nodes.size(2)
        gather_index = history_actions.reshape(batch_size, -1)[:, :, None].expand(-1, -1, embedding_dim)
        step_node_embeddings = self.encoded_nodes.gather(dim=1, index=gather_index)
        step_node_embeddings = step_node_embeddings.reshape(batch_size, rollout_size, seq_len, embedding_dim)
        return self.corrector(
            step_node_embeddings,
            history_step_features,
            history_valid_mask,
            removable_mask,
        )

    def score_macro_correction(
        self,
        history_actions,
        history_step_features,
        history_valid_mask,
        removable_mask,
        route_ids,
        route_valid_mask,
        route_features,
        global_features,
    ):
        if not self.use_learned_corrector or self.corrector is None or self.corrector_mode != 'macro_route_cut':
            raise RuntimeError("Macro corrector is disabled for this model.")

        if self.encoded_nodes is None:
            raise RuntimeError("encoded_nodes are not initialized. Call pre_forward first.")

        batch_size, rollout_size, seq_len = history_actions.shape
        embedding_dim = self.encoded_nodes.size(2)
        gather_index = history_actions.reshape(batch_size, -1)[:, :, None].expand(-1, -1, embedding_dim)
        step_node_embeddings = self.encoded_nodes.gather(dim=1, index=gather_index)
        step_node_embeddings = step_node_embeddings.reshape(batch_size, rollout_size, seq_len, embedding_dim)
        return self.corrector(
            step_node_embeddings,
            history_step_features,
            history_valid_mask,
            removable_mask,
            route_ids,
            route_valid_mask,
            route_features,
            global_features,
        )

def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, rollout)

    batch_size = node_index_to_pick.size(0)
    rollout_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, rollout_size, embedding_dim)
    # shape: (batch, rollout, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, rollout, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class CVRP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']
        self.use_depth_mixer = self.model_params.get('use_depth_mixer', False)
        node_static_dim = len(self.model_params.get('selected_node_static_feature_names', []))

        self.embedding_depot = nn.Linear(2, embedding_dim)
        node_feature_dim = 5 + node_static_dim
        if bool(self.model_params.get("include_service_duration_in_node_embedding", False)):
            node_feature_dim += 1
        self.embedding_node = nn.Linear(node_feature_dim, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])
        if self.use_depth_mixer:
            self.depth_key_norm = RMSNorm(embedding_dim)
            self.depth_queries = nn.Parameter(torch.empty(encoder_layer_num, embedding_dim))
            nn.init.normal_(self.depth_queries, mean=0.0, std=embedding_dim ** -0.5)

    def _mix_history(self, history_values, history_keys, query_vector):
        depth_scores = torch.einsum('bnle,e->bnl', history_keys, query_vector)
        depth_weights = torch.softmax(depth_scores, dim=2)
        return (depth_weights.unsqueeze(-1) * history_values).sum(dim=2)

    def forward(self, depot_xy, node_xy_demand):
        # depot_xy.shape: (batch, 1, 2)
        # node_xy_demand.shape: (batch, problem, 3)

        embedded_depot = self.embedding_depot(depot_xy)
        # shape: (batch, 1, embedding)
        embedded_node = self.embedding_node(node_xy_demand)
        # shape: (batch, problem, embedding)

        out = torch.cat((embedded_depot, embedded_node), dim=1)
        # shape: (batch, problem+1, embedding)

        if not self.use_depth_mixer:
            for layer in self.layers:
                out = layer(out)
            return out

        history_values = out.unsqueeze(2)
        history_keys = self.depth_key_norm(history_values)
        for layer_idx, layer in enumerate(self.layers):
            out = layer(
                self._mix_history(
                    history_values,
                    history_keys,
                    self.depth_queries[layer_idx],
                )
            )
            new_value = out.unsqueeze(2)
            history_values = torch.cat((history_values, new_value), dim=2)
            history_keys = torch.cat((history_keys, self.depth_key_norm(new_value)), dim=2)

        return out
        # shape: (batch, problem+1, embedding)


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        qkv_bias = bool(self.model_params.get('encoder_qkv_bias', False))
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=qkv_bias)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=qkv_bias)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=qkv_bias)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        self.use_gated_attention = self.model_params.get('use_gated_attention', False)
        self.alpha_attn_gate = float(self.model_params.get('alpha_attn_gate', 1.0))
        self.gated_attention_scale_mode = self.model_params.get('gated_attention_scale_mode', 'sigmoid')
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
        # input1.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        headwise_out = self.attention_fn(q, k, v, return_headwise=True)
        if self.use_gated_attention:
            gate_logits = self.W_gate(input1)
            if self.gated_attention_scale_mode == 'centered_sigmoid':
                gate = 2.0 * torch.sigmoid(gate_logits)
            else:
                gate = torch.sigmoid(gate_logits)
            gate = gate.transpose(1, 2).unsqueeze(-1)
            gated_headwise_out = headwise_out * gate.to(dtype=headwise_out.dtype)
            if self.alpha_attn_gate <= 0.0:
                pass
            elif self.alpha_attn_gate >= 1.0:
                headwise_out = gated_headwise_out
            else:
                headwise_out = (1.0 - self.alpha_attn_gate) * headwise_out + self.alpha_attn_gate * gated_headwise_out
        out_concat = reshape_from_heads(headwise_out)
        # shape: (batch, problem, head_num*qkv_dim)

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, embedding)

        out1 = self.add_n_normalization_1(input1, multi_head_out)
        return out1

    def forward_ffn(self, input1):
        out2 = self.feed_forward(input1)
        out3 = self.add_n_normalization_2(input1, out2)
        return out3

    def forward(self, input1):
        out1 = self.forward_attention(input1)
        out3 = self.forward_ffn(out1)

        return out3
        # shape: (batch, problem, embedding)


########################################
# DECODER
########################################

class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        poly_embedding_dim = self.model_params['poly_embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        z_dim = self.model_params['z_dim']
        self.use_decoder_checkpointing = bool(self.model_params.get('use_decoder_checkpointing', False))

        self.use_EAS_layers = False

        self.Wq_last = nn.Linear(embedding_dim+2, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.use_projected_logit_key = bool(self.model_params.get('use_projected_logit_key', False))
        self.Wlogit = nn.Linear(embedding_dim, embedding_dim, bias=False) if self.use_projected_logit_key else None

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.node_embeddings = None  # saved node embeddings for candidate feature bias
        self.z = None  # saved z vector for decoding
        self.use_poly_residual = bool(self.model_params.get('use_poly_residual', True))
        self.use_candidate_features = bool(self.model_params.get('use_candidate_features', False))
        self.candidate_rollout_chunk_size = max(1, int(self.model_params.get('candidate_rollout_chunk_size', 32)))
        selected_candidate_feature_names = list(self.model_params.get(
            'selected_candidate_feature_names',
            [],
        ))
        scorer_type = self.model_params.get('candidate_scorer_type', 'quotient_lite')
        if scorer_type not in {'baseline_additive', 'quotient_lite', 'mlp_score_only'}:
            raise ValueError(f"Unsupported candidate_scorer_type: {scorer_type}")
        self.candidate_scorer_type = scorer_type
        self.mlp_score_feature_mode = str(self.model_params.get('mlp_score_feature_mode', 'raw'))
        if self.mlp_score_feature_mode not in {'raw', 'centered'}:
            raise ValueError(f"Unsupported mlp_score_feature_mode: {self.mlp_score_feature_mode}")
        self.capture_candidate_aux = bool(self.model_params.get('capture_candidate_aux', False)) and scorer_type == 'quotient_lite'
        self.selected_candidate_feature_names = selected_candidate_feature_names
        self.selected_candidate_feature_indices = [
            CANDIDATE_FEATURE_INDEX[name] for name in selected_candidate_feature_names
        ]
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
            self.full_alpha_ctx_proj = nn.Linear(embedding_dim + 2, alpha_hidden, bias=False)
            self.full_alpha_node_proj = nn.Linear(embedding_dim, alpha_hidden, bias=True)
            self.full_alpha_out = nn.Linear(alpha_hidden, self.selected_candidate_feature_dim)
            self.full_phi_bias_norm = nn.LayerNorm(self.selected_candidate_feature_dim, elementwise_affine=False)
            self.full_phi_bias_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.full_gate_node = nn.Linear(embedding_dim, 1)
            self.full_gate_ctx = nn.Linear(embedding_dim + 2, 1)
            nn.init.zeros_(self.full_gate_node.weight)
            nn.init.zeros_(self.full_gate_node.bias)
            nn.init.zeros_(self.full_gate_ctx.weight)
            nn.init.constant_(self.full_gate_ctx.bias, float(self.model_params.get('candidate_full_gate_init_bias', 2.0)))
            self.full_gate_alpha = float(self.model_params.get('candidate_full_gate_alpha', 1.0))
            self.quotient_lite_candidate_scorer = None
            self.naive_feature_score_mlp = None
            if self.candidate_scorer_type == 'quotient_lite':
                self.quotient_lite_candidate_scorer = QuotientLiteCandidateScorer(
                    embedding_dim=embedding_dim,
                    feature_dim=self.selected_candidate_feature_dim,
                    selected_candidate_feature_names=self.selected_candidate_feature_names,
                    relative_candidate_feature_names=self.model_params.get('relative_candidate_feature_names', None),
                    zero_depot_relative_features=bool(self.model_params.get('zero_depot_relative_features', False)),
                    force_alpha_one=bool(self.model_params.get('qlite_force_alpha_one', False)),
                    disable_summary_modulation=bool(self.model_params.get('qlite_disable_summary_modulation', False)),
                    feature_centering_mode=self.model_params.get('qlite_feature_centering_mode', 'centered'),
                    summary_mode=self.model_params.get('qlite_summary_mode', 'partial'),
                    summary_dim=int(self.model_params.get('qlite_summary_dim', 4)),
                    hidden_dim=int(self.model_params.get('quotient_lite_hidden_dim', max(16, embedding_dim // 2))),
                    activation=self.model_params.get('quotient_scorer_activation', 'gelu'),
                    phi_proj_bias=self.model_params.get('phi_proj_bias', True),
                )
            elif self.candidate_scorer_type == 'mlp_score_only':
                hidden_dim = int(self.model_params.get(
                    'naive_feature_mlp_hidden_dim',
                    self.model_params.get('quotient_lite_hidden_dim', max(16, embedding_dim // 2)),
                ))
                activation = self.model_params.get('quotient_scorer_activation', 'gelu')
                self.naive_feature_score_mlp = nn.Sequential(
                    nn.Linear(embedding_dim + self.selected_candidate_feature_dim, hidden_dim),
                    _make_activation(activation),
                    nn.Linear(hidden_dim, 1),
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
            self.quotient_lite_candidate_scorer = None
            self.naive_feature_score_mlp = None


        if self.model_params['use_fast_attention']:
            self.attention_fn = fast_multi_head_attention
        else:
            self.attention_fn = multi_head_attention


    def set_kv(self, encoded_nodes, z=None):
        # encoded_nodes.shape: (batch, problem+1, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, problem+1, qkv_dim)
        logit_nodes = self.Wlogit(encoded_nodes) if self.Wlogit is not None else encoded_nodes
        self.single_head_key = logit_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem+1)
        self.node_embeddings = encoded_nodes
        # shape: (batch, problem+1, embedding)

        self.z = z
        # shape: (batch, rollout, z_dim)

    def set_z(self, z):
        self.z = z

    def _apply_shared_poly_residual(self, mh_atten_out):
        if not self.use_poly_residual:
            return torch.zeros_like(mh_atten_out)
        poly_out = self.poly_layer_1(torch.cat((mh_atten_out, self.z), dim=2))
        poly_out = F.relu(poly_out)
        return self.poly_layer_2(poly_out)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or rollout
        head_num = self.model_params['head_num']
        self.q1 = reshape_by_heads(self.Wq_1(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def set_q2(self, encoded_q2):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or rollout
        head_num = self.model_params['head_num']
        self.q2 = reshape_by_heads(self.Wq_2(encoded_q2), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def reset_EAS_layers(self, batch_size):
        self.EAS_W1 = torch.nn.Parameter(self.poly_layer_1.weight.mT.repeat(batch_size, 1, 1))
        self.EAS_b1 = torch.nn.Parameter(self.poly_layer_1.bias.repeat(batch_size, 1))
        self.EAS_W2 = torch.nn.Parameter(self.poly_layer_2.weight.mT.repeat(batch_size, 1, 1))
        self.EAS_b2 = torch.nn.Parameter(self.poly_layer_2.bias.repeat(batch_size, 1))
        self.use_EAS_layers = True

    def get_EAS_parameters(self):
        return [self.EAS_W1, self.EAS_b1, self.EAS_W2, self.EAS_b2]

    def _select_candidate_features(self, candidate_features):
        if candidate_features is None or not self.use_candidate_features:
            return None
        if getattr(candidate_features, "linc_cvrptw_feature_view", False):
            return None
        if candidate_features.size(-1) == self.selected_candidate_feature_dim:
            return candidate_features
        return candidate_features.index_select(dim=-1, index=self.selected_candidate_feature_index_tensor)

    def _build_baseline_gate_scalar(self, input_cat):
        if self.full_gate_node is None or self.full_gate_ctx is None:
            return None
        gate_node = self.full_gate_node(self.node_embeddings).squeeze(-1).unsqueeze(1)
        gate_ctx = self.full_gate_ctx(input_cat).squeeze(-1).unsqueeze(-1)
        return torch.sigmoid(gate_node + gate_ctx)

    def _apply_full_48_candidate_reasoning(self, mh_atten_out, input_cat, selected_phi, return_aux=False):
        rollout_size = mh_atten_out.size(1)
        score_static = torch.matmul(mh_atten_out, self.single_head_key)
        gate_node = self.full_gate_node(self.node_embeddings).squeeze(-1).unsqueeze(1)
        gate_ctx_all = self.full_gate_ctx(input_cat).squeeze(-1).unsqueeze(-1)
        phi_for_bias_all = self.full_phi_bias_norm(selected_phi)
        alpha_node_proj = self.full_alpha_node_proj(self.node_embeddings).unsqueeze(1)
        alpha_ctx_proj_all = self.full_alpha_ctx_proj(input_cat)
        phi_bias_scale = self.full_phi_bias_scale.to(dtype=mh_atten_out.dtype)
        phi_proj_is_linear = isinstance(self.full_phi_proj, nn.Linear)
        if phi_proj_is_linear:
            phi_proj_weight = self.full_phi_proj.weight
            phi_proj_bias = self.full_phi_proj.bias
        output_chunks = []
        static_chunks = []
        dynamic_chunks = []
        for start in range(0, rollout_size, self.candidate_rollout_chunk_size):
            end = min(start + self.candidate_rollout_chunk_size, rollout_size)
            mh_chunk = mh_atten_out[:, start:end, :]
            ctx_chunk = input_cat[:, start:end, :]
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
            if self.full_gate_alpha <= 0.0:
                mixed_score = score
            elif self.full_gate_alpha >= 1.0:
                mixed_score = score_gate
            else:
                mixed_score = (1.0 - self.full_gate_alpha) * score + self.full_gate_alpha * score_gate

            phi_for_bias = phi_for_bias_all[:, start:end, :, :]
            alpha_hidden = alpha_ctx_proj_all[:, start:end, :].unsqueeze(2) + alpha_node_proj
            alpha_logits = self.full_alpha_out(F.relu(alpha_hidden))
            alpha_weights = F.softmax(alpha_logits, dim=-1)
            phi_bias = (alpha_weights * phi_for_bias).sum(dim=-1)
            final_chunk = mixed_score + phi_bias_scale * phi_bias
            output_chunks.append(final_chunk)
            if return_aux:
                static_chunk = score_static[:, start:end, :]
                static_chunks.append(static_chunk)
                dynamic_chunks.append(final_chunk - static_chunk)
        score_total = torch.cat(output_chunks, dim=1)
        if not return_aux:
            return score_total
        return score_total, {
            "baseline_base_score": torch.cat(static_chunks, dim=1),
            "baseline_dynamic_score": torch.cat(dynamic_chunks, dim=1),
        }

    def _apply_quotient_lite_reasoning(self, mh_atten_out, selected_phi, candidate_features, ninf_mask, return_aux=False):
        rollout_size = mh_atten_out.size(1)
        chunk_size = self.candidate_rollout_chunk_size
        can_chunk = (
            not return_aux
            and chunk_size > 0
            and chunk_size < rollout_size
            and selected_phi is not None
            and torch.is_tensor(candidate_features)
        )
        if can_chunk:
            output_chunks = []
            for start in range(0, rollout_size, chunk_size):
                end = min(start + chunk_size, rollout_size)
                score, _ = self.quotient_lite_candidate_scorer(
                    decoder_context=mh_atten_out[:, start:end, :],
                    candidate_key=self.node_embeddings,
                    candidate_features=candidate_features[:, start:end, :, :],
                    ninf_mask=ninf_mask[:, start:end, :],
                    selected_phi=selected_phi[:, start:end, :, :],
                    return_aux=False,
                )
                output_chunks.append(score)
            return torch.cat(output_chunks, dim=1)

        score, scorer_aux = self.quotient_lite_candidate_scorer(
            decoder_context=mh_atten_out,
            candidate_key=self.node_embeddings,
            candidate_features=candidate_features,
            ninf_mask=ninf_mask,
            selected_phi=selected_phi,
            return_aux=return_aux,
        )
        if return_aux:
            return score, scorer_aux
        return score

    def _center_selected_candidate_features(self, selected_phi, ninf_mask):
        feasible = torch.isfinite(ninf_mask[:, :, 1:]).to(dtype=selected_phi.dtype)
        denom = feasible.sum(dim=2, keepdim=True).clamp_min(1.0)
        customer_phi = selected_phi[:, :, 1:, :]
        mean = (customer_phi * feasible.unsqueeze(-1)).sum(dim=2, keepdim=True) / denom.unsqueeze(-1)
        centered = selected_phi.clone()
        centered[:, :, 1:, :] = customer_phi - mean
        centered[:, :, 0, :] = 0
        return centered

    def _apply_mlp_score_only_reasoning(self, mh_atten_out, selected_phi, ninf_mask):
        rollout_size = mh_atten_out.size(1)
        base_score = torch.matmul(mh_atten_out, self.single_head_key)
        output_chunks = []
        for start in range(0, rollout_size, self.candidate_rollout_chunk_size):
            end = min(start + self.candidate_rollout_chunk_size, rollout_size)
            ctx_chunk = mh_atten_out[:, start:end, :]
            phi_chunk = selected_phi[:, start:end, :, :].to(dtype=ctx_chunk.dtype)
            if self.mlp_score_feature_mode == 'centered':
                phi_chunk = self._center_selected_candidate_features(phi_chunk, ninf_mask[:, start:end, :])
            ctx_expanded = ctx_chunk[:, :, None, :].expand(-1, -1, phi_chunk.size(2), -1)
            feature_score = self.naive_feature_score_mlp(torch.cat((ctx_expanded, phi_chunk), dim=-1)).squeeze(-1)
            output_chunks.append(base_score[:, start:end, :] + feature_score)
        return torch.cat(output_chunks, dim=1)

    def forward(
        self,
        encoded_last_node,
        load,
        time,
        ninf_mask,
        candidate_features=None,
        current_node=None,
        visited_mask=None,
        finished=None,
        return_logits=False,
    ):
        # encoded_last_node.shape: (batch, rollout, embedding)
        # load.shape: (batch, rollout)
        # ninf_mask.shape: (batch, rollout, problem)

        head_num = self.model_params['head_num']
        self.last_forward_aux = None

        #  Multi-Head Attention
        #######################################################
        input_cat = torch.cat((encoded_last_node, load[:, :, None], time[:, :, None]), dim=2)
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(input_cat), head_num=head_num)
        # shape: (batch, head_num, rollout, qkv_dim)

        q = q_last
        # shape: (batch, head_num, rollout, qkv_dim)

        selected_phi = None
        has_candidate_features = candidate_features is not None and self.use_candidate_features
        if has_candidate_features:
            selected_phi = self._select_candidate_features(candidate_features)
            if selected_phi is not None:
                selected_phi = selected_phi.to(dtype=q.dtype)

        out_concat = self.attention_fn(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, rollout, head_num*qkv_dim)

        mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, rollout, embedding)

        #####  PolyNet Layers Start ##############

        if not self.use_EAS_layers:
            #  PolyNet without EAS (default)

            poly_out = self._apply_shared_poly_residual(mh_atten_out)
            # shape: (batch, rollout, poly_embedding_dim)
        else:
            if not self.use_poly_residual:
                raise RuntimeError("EAS requires PolyNet residual layers, but use_poly_residual=False")
            # PolyNet with EAS

            poly_out = torch.matmul(torch.cat((mh_atten_out, self.z), dim=2), self.EAS_W1)
            # shape: (batch, rollout, poly_embedding_dim)
            poly_out += self.EAS_b1[:, None]
            # shape: (batch, rollout, poly_embedding_dim)
            poly_out = F.relu(poly_out)
            # shape: (batch, rollout, poly_embedding_dim)
            poly_out = torch.matmul(poly_out, self.EAS_W2)
            # shape: (batch, rollout, poly_embedding_dim)
            poly_out += self.EAS_b2[:, None]
            # shape: (batch, rollout, poly_embedding_dim)

        #####  PolyNet Layers End ##############

        mh_atten_out = mh_atten_out + poly_out

        #  Single-Head Attention, for probability calculation
        #######################################################
        scorer_aux = None
        baseline_gate_scalar = None
        if has_candidate_features:
            if self.candidate_scorer_type == 'quotient_lite':
                if self.use_decoder_checkpointing and self.training:
                    score = checkpoint(
                        lambda a, b, c, d: self._apply_quotient_lite_reasoning(a, b, c, d, return_aux=False),
                        mh_atten_out,
                        selected_phi,
                        candidate_features,
                        ninf_mask,
                        use_reentrant=False,
                    )
                    if self.capture_candidate_aux:
                        with torch.no_grad():
                            _, scorer_aux = self._apply_quotient_lite_reasoning(
                                mh_atten_out.detach(),
                                selected_phi.detach(),
                                candidate_features.detach(),
                                ninf_mask.detach(),
                                return_aux=True,
                            )
                else:
                    if not self.capture_candidate_aux:
                        score = self._apply_quotient_lite_reasoning(
                            mh_atten_out,
                            selected_phi,
                            candidate_features,
                            ninf_mask,
                            return_aux=False,
                        )
                    else:
                        score, scorer_aux = self._apply_quotient_lite_reasoning(
                            mh_atten_out,
                            selected_phi,
                            candidate_features,
                            ninf_mask,
                            return_aux=True,
                        )
            elif self.candidate_scorer_type == 'mlp_score_only':
                if selected_phi is None:
                    raise RuntimeError("Fused CVRPTW candidate feature views require quotient_lite scoring.")
                if self.use_decoder_checkpointing and self.training:
                    score = checkpoint(
                        lambda a, b, c: self._apply_mlp_score_only_reasoning(a, b, c),
                        mh_atten_out,
                        selected_phi,
                        ninf_mask,
                        use_reentrant=False,
                    )
                else:
                    score = self._apply_mlp_score_only_reasoning(mh_atten_out, selected_phi, ninf_mask)
            elif self.use_decoder_checkpointing and self.training:
                if selected_phi is None:
                    raise RuntimeError("Fused CVRPTW candidate feature views require quotient_lite scoring.")
                score = checkpoint(
                    lambda a, b, c: self._apply_full_48_candidate_reasoning(a, b, c),
                    mh_atten_out,
                    input_cat,
                    selected_phi,
                    use_reentrant=False,
                )
            else:
                if selected_phi is None:
                    raise RuntimeError("Fused CVRPTW candidate feature views require quotient_lite scoring.")
                score = self._apply_full_48_candidate_reasoning(mh_atten_out, input_cat, selected_phi)
        else:
            score = torch.matmul(mh_atten_out, self.single_head_key)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, rollout, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        if return_logits:
            return score_masked

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, rollout, problem)

        if has_candidate_features and baseline_gate_scalar is None and self.capture_candidate_aux:
            baseline_gate_scalar = self._build_baseline_gate_scalar(input_cat)

        if scorer_aux is not None:
            aux = {}
            aux.update(scorer_aux)
            gate_tensor = None
            if baseline_gate_scalar is not None:
                gate_tensor = baseline_gate_scalar
            if gate_tensor is not None:
                aux['gate_mean'] = gate_tensor.mean()
                aux['gate_std'] = gate_tensor.std(unbiased=False)
            aux['probs'] = probs
            self.last_forward_aux = aux

        return probs


class CVRPTWCorrector(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = int(model_params['embedding_dim'])
        hidden_dim = int(model_params.get('corrector_hidden_dim', embedding_dim))
        step_feature_dim = int(model_params.get('corrector_step_feature_dim', len(CANDIDATE_FEATURE_INDEX) + 5))
        max_removals = max(1, int(model_params.get('corrector_max_removals', 4)))

        self.max_removals = max_removals
        self.step_feature_dim = step_feature_dim

        self.step_encoder = nn.Sequential(
            nn.Linear(embedding_dim + step_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)
        self.local_proj = nn.Linear(hidden_dim, hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_removals + 1),
        )

    def forward(self, step_node_embeddings, history_step_features, history_valid_mask, removable_mask):
        history_valid_mask = history_valid_mask.to(dtype=torch.bool)
        removable_mask = removable_mask.to(dtype=torch.bool)

        step_input = torch.cat((step_node_embeddings, history_step_features), dim=-1)
        step_hidden = self.step_encoder(step_input)

        valid_float = history_valid_mask.to(dtype=step_hidden.dtype)
        denom = valid_float.sum(dim=2, keepdim=True).clamp_min(1.0)
        pooled = (step_hidden * valid_float.unsqueeze(-1)).sum(dim=2) / denom
        global_ctx = self.global_proj(pooled).unsqueeze(2)
        local_ctx = self.local_proj(step_hidden)
        step_logits = self.score_head(F.relu(local_ctx + global_ctx)).squeeze(-1)
        step_logits = step_logits.masked_fill(~removable_mask, float('-inf'))

        removable_float = removable_mask.to(dtype=step_hidden.dtype)
        removable_hidden = (step_hidden * removable_float.unsqueeze(-1)).sum(dim=2) / removable_float.sum(dim=2, keepdim=True).clamp_min(1.0)
        count_logits = self.count_head(removable_hidden)
        return count_logits, step_logits


class CVRPTWMacroCorrector(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = int(model_params['embedding_dim'])
        hidden_dim = int(model_params.get('corrector_hidden_dim', embedding_dim))
        step_feature_dim = int(model_params.get('corrector_step_feature_dim', len(CANDIDATE_FEATURE_INDEX) + 5))

        self.step_encoder = nn.Sequential(
            nn.Linear(embedding_dim + step_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.route_encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.route_head = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.cut_encoder = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cut_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        step_node_embeddings,
        history_step_features,
        history_valid_mask,
        removable_mask,
        route_ids,
        route_valid_mask,
        route_features,
        global_features,
    ):
        history_valid_mask = history_valid_mask.to(dtype=torch.bool)
        removable_mask = removable_mask.to(dtype=torch.bool)
        route_valid_mask = route_valid_mask.to(dtype=torch.bool)

        step_input = torch.cat((step_node_embeddings, history_step_features), dim=-1)
        step_hidden = self.step_encoder(step_input)

        batch_size, rollout_size, seq_len, hidden_dim = step_hidden.shape
        max_routes = route_valid_mask.size(2)
        route_hidden_seed = torch.zeros(
            (batch_size, rollout_size, max_routes, hidden_dim),
            dtype=step_hidden.dtype,
            device=step_hidden.device,
        )

        for route_idx in range(max_routes):
            route_mask = history_valid_mask & route_ids.eq(route_idx)
            route_float = route_mask.to(dtype=step_hidden.dtype).unsqueeze(-1)
            route_sum = (step_hidden * route_float).sum(dim=2)
            route_count = route_float.sum(dim=2).clamp_min(1.0)
            route_mean = route_sum / route_count

            masked_hidden = step_hidden.masked_fill(~route_mask.unsqueeze(-1), float('-inf'))
            route_max = masked_hidden.max(dim=2).values
            route_max = torch.where(torch.isfinite(route_max), route_max, torch.zeros_like(route_max))
            route_hidden_seed[:, :, route_idx, :] = 0.5 * (route_mean + route_max)

        global_expand = global_features[:, :, None, :].expand(-1, -1, max_routes, -1)
        route_input = torch.cat((route_hidden_seed, route_features, global_expand), dim=-1)
        route_hidden = self.route_encoder(route_input)
        route_logits = self.route_head(route_hidden).squeeze(-1)
        route_logits = route_logits.masked_fill(~route_valid_mask, float('-inf'))

        route_valid_float = route_valid_mask.to(dtype=step_hidden.dtype).unsqueeze(-1)
        route_pooled = (route_hidden * route_valid_float).sum(dim=2) / route_valid_float.sum(dim=2).clamp_min(1.0)
        count_input = torch.cat((route_pooled, global_features), dim=-1)
        count_logits = self.count_head(count_input)

        route_ctx_per_step = torch.zeros_like(step_hidden)
        for route_idx in range(max_routes):
            route_mask = route_ids.eq(route_idx).unsqueeze(-1).to(dtype=step_hidden.dtype)
            route_ctx_per_step = route_ctx_per_step + route_hidden[:, :, route_idx, :].unsqueeze(2) * route_mask

        cut_input = torch.cat(
            (
                step_hidden,
                route_ctx_per_step,
                global_features[:, :, None, :].expand(-1, -1, seq_len, -1),
            ),
            dim=-1,
        )
        cut_hidden = self.cut_encoder(cut_input)
        cut_logits = self.cut_head(cut_hidden).squeeze(-1)
        cut_logits = cut_logits.masked_fill(~removable_mask, float('-inf'))
        return count_logits, route_logits, cut_logits


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def reshape_from_heads(headwise_out):
    out_transposed = headwise_out.transpose(1, 2)
    batch_s = out_transposed.size(0)
    n = out_transposed.size(1)
    head_num = out_transposed.size(2)
    key_dim = out_transposed.size(3)
    return out_transposed.reshape(batch_s, n, head_num * key_dim)


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None, return_headwise=False):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    if return_headwise:
        return out

    return reshape_from_heads(out)


def fast_multi_head_attention(q, k, v, rank3_ninf_mask=None, return_headwise=False):
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    mask = None
    if rank3_ninf_mask is not None:
        mask = rank3_ninf_mask[:, None, :, :]
        mask = mask.expand(batch_s, head_num, n, input_s)

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    if return_headwise:
        return out

    return reshape_from_heads(out)


class AddAndInstanceNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class AddAndBatchNormalization(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm_by_EMB = nn.BatchNorm1d(embedding_dim, affine=True)
        # 'Funny' Batch_Norm, as it will normalized by EMB dim

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        batch_s = input1.size(0)
        problem_s = input1.size(1)
        embedding_dim = input1.size(2)

        added = input1 + input2
        normalized = self.norm_by_EMB(added.reshape(batch_s * problem_s, embedding_dim))
        back_trans = normalized.reshape(batch_s, problem_s, embedding_dim)

        return back_trans

class FeedForward(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


class RMSNorm(nn.Module):
    def __init__(self, embedding_dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight
