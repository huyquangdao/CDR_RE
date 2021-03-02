import torch
import torch.nn as nn


class GraphEncoder(nn.Module):
    def __init__(
        self,
        time_step,
        word_vocab_size,
        egde_vocab_size,
        pos_vocab_size,
        char_vocab_size,
        hypernym_vocab_size,
        synonym_vocab_size,
        word_embedding_dim,
        edge_embedding_dim,
        pos_embedding_dim,
        combined_embedding_dim,
        transformer_attn_head,
        transformer_block,
        kernel_size=5,
        n_filters=30,
        max_seq_length=700,
        encoder_hidden_size=150,
        char_embedding_dim=30,
        hypernym_embedding_dim=10,
        synonym_embedding_dim=10,
        use_transformer=True,
        use_self_atentive=False,
        device=torch.device("cuda:0"),
        drop_out=0.2,
    ):

        super(GraphEncoder, self).__init__()

        self.word_embedding = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.in_edge_embedding = nn.Embedding(egde_vocab_size, edge_embedding_dim)
        self.out_edge_embedding = nn.Embedding(egde_vocab_size, edge_embedding_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)

        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim)
        self.hypernym_embedding = nn.Embedding(hypernym_vocab_size, hypernym_embedding_dim)
        self.synonym_embedding = nn.Embedding(synonym_vocab_size, synonym_embedding_dim)

        self.n_filters = n_filters

        # position_embedding_dim = 50
        # self.position_embedding = nn.Embedding(max_seq_length, position_embedding_dim)

        self.conv = nn.Conv1d(
            in_channels=char_embedding_dim, out_channels=n_filters, kernel_size=kernel_size, stride=1
        )

        # torch.nn.init.xavier_normal(self.in_edge_embedding.weight)
        # torch.nn.init.xavier_normal(self.pos_embedding.weight)
        # torch.nn.init.xavier_normal(self.out_edge_embedding.weight)

        self.drop_out = nn.Dropout(drop_out)

        self.char_embedding_dim = char_embedding_dim
        self.linear_node_edge = nn.Linear(
            word_embedding_dim + edge_embedding_dim + pos_embedding_dim + char_embedding_dim, combined_embedding_dim
        )

        self.time_step = time_step
        self.device = device

        self.encoder_hidden_size = encoder_hidden_size
        self.word_embedding_dim = word_embedding_dim
        self.combined_embedding_dim = combined_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.pos_embedding_dim = pos_embedding_dim
        self.synonym_embedding_dim = synonym_embedding_dim
        self.hypernym_embedding_dim = hypernym_embedding_dim

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=combined_embedding_dim, nhead=transformer_attn_head, dim_feedforward=1024
        )
        self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=transformer_block)

        self.use_transformer = use_transformer

        self.w_h = nn.Linear(self.word_embedding_dim, self.encoder_hidden_size)
        self.w_cell = nn.Linear(self.word_embedding_dim, self.encoder_hidden_size)

        self.w_in_ingate = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size, bias=False)
        self.u_in_ingate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_ingate = nn.Parameter(torch.zeros(self.encoder_hidden_size))
        self.w_out_ingate = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size, bias=False)
        self.u_out_ingate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        # weight for attn
        self.k_in_ingate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.k_out_ingate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        self.w_in_forgetgate = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size, bias=False)
        self.u_in_forgetgate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_forgetgate = nn.Parameter(torch.zeros(self.encoder_hidden_size))
        self.w_out_forgetgate = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size, bias=False)
        self.u_out_forgetgate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        # weight for attn
        self.k_in_forgetgate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.k_out_forgetgate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        self.w_in_outgate = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size, bias=False)
        self.u_in_outgate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_outgate = nn.Parameter(torch.zeros(self.encoder_hidden_size))
        self.w_out_outgate = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size, bias=False)
        self.u_out_outgate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        # weight for attn
        self.k_in_outgate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.k_out_outgate = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        self.w_in_cell = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size, bias=False)
        self.u_in_cell = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_cell = nn.Parameter(torch.zeros(self.encoder_hidden_size))
        self.w_out_cell = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size, bias=False)
        self.u_out_cell = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        # weight for attn
        self.k_in_cell = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.k_out_cell = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        # weight for attn
        self.W_g_in = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.U_g_in = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_g_in = nn.Parameter(torch.zeros(self.encoder_hidden_size))
        self.W_f_in = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.U_f_in = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_f_in = nn.Parameter(torch.zeros(self.encoder_hidden_size))

        self.W_o_in = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.U_o_in = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_o_in = nn.Parameter(torch.zeros(self.encoder_hidden_size))

        self.W_g_out = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.U_g_out = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_g_out = nn.Parameter(torch.zeros(self.encoder_hidden_size))
        self.W_f_out = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.U_f_out = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_f_out = nn.Parameter(torch.zeros(self.encoder_hidden_size))

        self.W_o_out = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.U_o_out = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)
        self.b_o_out = nn.Parameter(torch.zeros(self.encoder_hidden_size))

        self.W_in = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size)
        self.W_out = nn.Linear(self.combined_embedding_dim, self.encoder_hidden_size)

        self.use_self_atentive = False

        # for span attention
        self.W_spans_in = nn.Parameter(torch.randn(2, self.encoder_hidden_size))
        self.ffn_alpha_in = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

        self.W_spans_out = nn.Parameter(torch.randn(10, self.encoder_hidden_size))
        self.ffn_alpha_out = nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size, bias=False)

    # def cal_span_representation(self, collected_hidden_states, W_spans, ffn_alpha, mask = None):

    #     scores = torch.tanh(ffn_alpha(collected_hidden_states))

    #     #scores = [b, max_seq_length, max_spans, hidden_size]

    #     scores = torch.matmul(W_spans, scores.permute(0,1,3,2)).squeeze(2)

    #     #scores = [b, max_seq_length, max_spans]

    #     scores = scores.masked_fill(mask == 0, -1e10)

    #     scores = torch.softmax(scores, dim =2)

    #     #scores = [b, max_seq_length, max_spans]

    #     span_representation = torch.matmul(collected_hidden_states.permute(0,1,3,2), scores.unsqueeze(-1)).squeeze(-1)

    #     return span_representation

    def cal_span_representation(self, collected_hidden_states, W_spans, ffn_alpha, mask=None):

        scores = torch.tanh(ffn_alpha(collected_hidden_states))

        # scores = [b, max_seq_length, max_spans, hidden_size]

        scores = torch.matmul(W_spans, scores.permute(0, 1, 3, 2))

        # scores = [b, max_seq_length, 4, max_spans]

        # print(scores.shape)
        # print(mask.shape)

        scores = scores.masked_fill(mask.unsqueeze(-2) == 0, -1e10)

        scores = torch.softmax(scores, dim=-1)

        # scores = [b, max_seq_length, 4, max_spans]

        span_representation = torch.matmul(collected_hidden_states.permute(0, 1, 3, 2), scores.permute(0, 1, 3, 2))

        # span_representation = [b, max_seq_length, hidden_size, 4]

        span_representation = torch.sum(span_representation, dim=-1)

        return span_representation

    def collect_neighbor_representations(self, representations, positions):

        # representation = [batch, max_seg_length, hidden_dim]
        # positions = [batch, max_seq_length, max_node_to_collect]

        batch, max_seq_length, max_node_to_collect = positions.shape
        feature_dim = representations.shape[-1]

        positions = positions.view(batch, max_seq_length * max_node_to_collect)

        # positions = [batch, max_seq_length * max_node_to_collect ]

        collected_tensor = torch.gather(
            representations, 1, positions[..., None].expand(*positions.shape, representations.shape[-1])
        )

        collected_tensor = collected_tensor.view(batch, max_seq_length, max_node_to_collect, feature_dim)

        return collected_tensor

    def calculate_attn_context(self, g_in, g_out, cell, in_embedded, out_embedded, tokens_mask=None):

        bs, max_seq_length, _ = in_embedded.shape

        max_in_hidden = in_embedded.max(dim=1)[0]
        max_out_hidden = out_embedded.max(dim=1)[0]

        f_g_in = torch.sigmoid(self.W_g_in(g_in) + self.U_g_in(max_in_hidden) + self.b_g_in)
        # [b,h]

        f_in = torch.sigmoid(
            self.W_f_in(g_in.unsqueeze(1).repeat(1, max_seq_length, 1)) + self.U_f_in(in_embedded) + self.b_f_in
        )
        # f_in = [b,s,h]

        o_in = torch.sigmoid(self.W_o_in(g_in) + self.U_o_in(max_in_hidden) + self.b_o_in)

        tmp_in = torch.softmax(
            torch.cat([f_in.masked_fill(tokens_mask.unsqueeze(-1) == 0, -1e10), f_g_in.unsqueeze(1)], dim=1), dim=1
        )
        # [b,m+1,h]

        f_in = tmp_in[:, :-1, :]
        f_g_in = tmp_in[:, -1, :]

        # cell =[b,s,h]

        attn_cell_in = torch.sum((f_in * cell) * tokens_mask.unsqueeze(-1), dim=1) + f_g_in * g_in
        g_in = o_in * torch.tanh(attn_cell_in)

        f_g_out = torch.sigmoid(self.W_g_out(g_out) + self.U_g_out(max_out_hidden) + self.b_g_out)
        # [b,h]

        f_out = torch.sigmoid(
            self.W_f_out(g_out.unsqueeze(1).repeat(1, max_seq_length, 1)) + self.U_f_out(out_embedded) + self.b_f_out
        )
        # f_in = [b,s,h]

        o_out = torch.sigmoid(self.W_o_out(g_out) + self.U_o_out(max_out_hidden) + self.b_o_out)

        tmp_out = torch.softmax(
            torch.cat([f_out.masked_fill(tokens_mask.unsqueeze(-1) == 0, -1e10), f_g_out.unsqueeze(1)], dim=1), dim=1
        )
        # [b,m+1,h]

        f_out = tmp_out[:, :-1, :]
        f_g_out = tmp_out[:, -1, :]

        # cell =[b,s,h]

        attn_cell_out = torch.sum((f_out * cell) * tokens_mask.unsqueeze(-1), dim=1) + f_g_out * g_out

        g_out = o_out * torch.tanh(attn_cell_out)

        return g_in, g_out

    def forward(self, inputs):

        (
            token_ids_tensor,
            token_ids_mask_tensor,
            pos_ids_tensor,
            char_ids_tensor,
            hypernym_ids_tensor,
            synonym_ids_tensor,
            in_nodes_idx_tensor,
            in_nodes_mask_tensor,
            out_nodes_idx_tensor,
            out_nodes_mask_tensor,
            in_edge_idx_tensor,
            in_edge_idx_mask,
            out_edge_idx_tensor,
            out_edge_idx_mask,
        ) = inputs

        bs, batch_length = token_ids_tensor.shape
        max_char_length = char_ids_tensor.shape[2]

        # batch_size, max_seq_length, word_embedding_dim
        word_embedded = self.word_embedding(token_ids_tensor)

        # batch_size, max_seq_length, word_embedding_dim
        word_embedded = token_ids_mask_tensor.unsqueeze(-1) * word_embedded

        # drop out word embedded
        # word_embedded = self.drop_out(word_embedded)

        char_embedded = self.char_embedding(char_ids_tensor)

        # char_embedded = [batch_size, max_seq_length, max_char_length, char_embedding_dim]

        char_embedded = char_embedded.view(bs * batch_length, max_char_length, self.char_embedding_dim)

        conv_char_embedded = torch.tanh(self.conv(char_embedded.permute(0, 2, 1)))

        # drop out char conv
        # conv_char_embedded = self.drop_out(conv_char_embedded)

        # conv_char_embedded = [bs * max_seq, char_embedding_dim, max_char_length]

        conv_char_embedded = torch.max(conv_char_embedded.permute(0, 2, 1), dim=1)[0]

        # conv_char_embedded = [bs * max_seq, char_embedding_dim]

        conv_char_embedded = conv_char_embedded.view(bs, batch_length, self.n_filters)

        pos_embedded = self.pos_embedding(pos_ids_tensor)
        hypernym_embedded = self.hypernym_embedding(hypernym_ids_tensor)
        synonym_embedded = self.synonym_embedding(synonym_ids_tensor)

        concat_word_pos_embedded = self.drop_out(torch.cat([word_embedded, pos_embedded, conv_char_embedded], dim=-1))

        # batch_size, max_seq_length, max_node_in, edge_embedding_dim
        in_edge_embedded = self.in_edge_embedding(in_edge_idx_tensor)

        # batch_size, max_seq_length, max_node_in, edge_embedding_dim + word_embedding_dim
        collected_in_word_embedded = self.collect_neighbor_representations(
            concat_word_pos_embedded, in_nodes_idx_tensor
        )
        in_embedded = torch.cat([collected_in_word_embedded, in_edge_embedded], dim=-1)

        # multiply with mask
        in_embedded = in_embedded * in_nodes_mask_tensor.unsqueeze(-1)
        # sum over dimension 2

        in_embedded = torch.sum(in_embedded, dim=2)

        # in_embedded = self.cal_span_representation(in_embedded, in_nodes_mask_tensor)

        # batch_size, max_seq_length, max_node_in, edge_embedding_dim
        out_edge_embedded = self.out_edge_embedding(out_edge_idx_tensor)

        # batch_size, max_seq_length, max_node_in, edge_embedding_dim + word_embedding_dim
        collected_out_word_embedded = self.collect_neighbor_representations(
            concat_word_pos_embedded, out_nodes_idx_tensor
        )
        out_embedded = torch.cat([collected_out_word_embedded, out_edge_embedded], dim=-1)

        # multiply with mask
        out_embedded = out_embedded * out_nodes_mask_tensor.unsqueeze(-1)

        # sum over dimension 2
        out_embedded = torch.sum(out_embedded, dim=2)

        # in_embedded = self.bn_w(in_embedded.permute(0,2,1)).permute(0,2,1)
        # out_embedded = self.bn_w(out_embedded.permute(0,2,1)).permute(0,2,1)

        # project to lower dimension and apply non linear function
        in_embedded = torch.tanh(self.linear_node_edge(in_embedded))
        out_embedded = torch.tanh(self.linear_node_edge(out_embedded))

        bs, max_seq_length, _ = in_embedded.shape

        # node_hidden = torch.zeros(size=(bs, max_seq_length, self.hidden_size)).to(self.device)
        # node_cell = torch.zeros(size=(bs, max_seq_length, self.hidden_size)).to(self.device)

        node_hidden = torch.tanh(self.w_h(word_embedded))
        node_cell = torch.tanh(self.w_cell(word_embedded))

        if self.use_transformer:
            in_embedded = self.transformer(
                in_embedded.permute(1, 0, 2),
                src_key_padding_mask=(1 - token_ids_mask_tensor).type(torch.cuda.BoolTensor),
            )
            out_embedded = self.transformer(
                out_embedded.permute(1, 0, 2),
                src_key_padding_mask=(1 - token_ids_mask_tensor).type(torch.cuda.BoolTensor),
            )
            in_embedded = in_embedded.permute(1, 0, 2)
            out_embedded = out_embedded.permute(1, 0, 2)

        g_in = torch.zeros(size=(bs, self.hidden_size)).to(self.device)
        g_out = torch.zeros(size=(bs, self.hidden_size)).to(self.device)

        for t in range(self.time_step):

            passage_in_edge_prev_hidden = self.collect_neighbor_representations(node_hidden, in_nodes_idx_tensor)
            passage_out_edge_prev_hidden = self.collect_neighbor_representations(node_hidden, out_nodes_idx_tensor)

            if self.use_self_atentive:
                passage_in_edge_prev_hidden = self.cal_span_representation(
                    passage_in_edge_prev_hidden, self.W_spans_in, self.ffn_alpha_in, in_nodes_mask_tensor
                )
                passage_in_edge_prev_hidden = passage_in_edge_prev_hidden * token_ids_mask_tensor.unsqueeze(-1)

                passage_out_edge_prev_hidden = self.cal_span_representation(
                    passage_out_edge_prev_hidden, self.W_spans_out, self.ffn_alpha_out, out_nodes_mask_tensor
                )
                passage_out_edge_prev_hidden = passage_out_edge_prev_hidden * token_ids_mask_tensor.unsqueeze(-1)

            else:
                passage_in_edge_prev_hidden = passage_in_edge_prev_hidden * in_nodes_mask_tensor.unsqueeze(-1)
                # [batch_size, node_len, neighbor_vector_dim]
                passage_in_edge_prev_hidden = torch.sum(passage_in_edge_prev_hidden, dim=2)
                passage_out_edge_prev_hidden = passage_out_edge_prev_hidden * out_nodes_mask_tensor.unsqueeze(-1)
                # [batch_size, node_len, neighbor_vector_dim]
                passage_out_edge_prev_hidden = torch.sum(passage_out_edge_prev_hidden, dim=2)

            passage_edge_ingate = torch.sigmoid(
                self.w_in_ingate(in_embedded)
                + self.u_in_ingate(passage_in_edge_prev_hidden)
                + self.w_out_ingate(out_embedded)
                + self.u_out_ingate(passage_out_edge_prev_hidden)
                # + self.k_in_ingate(g_in.unsqueeze(1).repeat(1,max_seq_length,1))
                # + self.k_out_ingate(g_out.unsqueeze(1).repeat(1,max_seq_length,1))
                + self.b_ingate
            )

            passage_edge_forgetgate = torch.sigmoid(
                self.w_in_forgetgate(in_embedded)
                + self.u_in_forgetgate(passage_in_edge_prev_hidden)
                + self.w_out_forgetgate(out_embedded)
                + self.u_out_forgetgate(passage_out_edge_prev_hidden)
                # + self.k_in_forgetgate(g_in.unsqueeze(1).repeat(1,max_seq_length,1))
                # + self.k_out_forgetgate(g_out.unsqueeze(1).repeat(1,max_seq_length,1))
                + self.b_forgetgate
            )

            passage_edge_outgate = torch.sigmoid(
                self.w_in_outgate(in_embedded)
                + self.u_in_outgate(passage_in_edge_prev_hidden)
                + self.w_out_outgate(out_embedded)
                + self.u_out_outgate(passage_out_edge_prev_hidden)
                # + self.k_in_outgate(g_in.unsqueeze(1).repeat(1,max_seq_length,1))
                # + self.k_out_outgate(g_out.unsqueeze(1).repeat(1,max_seq_length,1))
                + self.b_outgate
            )

            passage_edge_cell_input = torch.sigmoid(
                self.w_in_cell(in_embedded)
                + self.u_in_cell(passage_in_edge_prev_hidden)
                + self.w_out_cell(out_embedded)
                + self.u_out_cell(passage_out_edge_prev_hidden)
                # + self.k_in_cell(g_in.unsqueeze(1).repeat(1,max_seq_length,1))
                # + self.k_out_cell(g_out.unsqueeze(1).repeat(1,max_seq_length,1))
                + self.b_cell
            )

            passage_edge_cell = passage_edge_forgetgate * node_cell + passage_edge_ingate * passage_edge_cell_input
            passage_edge_hidden = passage_edge_outgate * torch.tanh(passage_edge_cell)
            # node mask
            # [batch_size, passage_len, neighbor_vector_dim]
            node_cell = passage_edge_cell * token_ids_mask_tensor.unsqueeze(-1)
            node_hidden = passage_edge_hidden * token_ids_mask_tensor.unsqueeze(-1)

            # g_in, g_out = self.calculate_attn_context(g_in,
            #                                           g_out,
            #                                           node_cell,
            #                                           passage_in_edge_prev_hidden,
            #                                           passage_out_edge_prev_hidden,
            #                                           token_ids_mask_tensor)

        return node_cell, node_hidden, concat_word_pos_embedded


class GraphStateLSTM(nn.Module):
    def __init__(
        self,
        relation_classes,
        ner_classes,
        encoder,
        glstm_hidden_size,
        elmo_hidden_size,
        flair_hidden_size,
        max_distant,
        distant_embedding_dim,
        ner_hidden_size,
        lstm_layers,
        use_ner=False,
        drop_out=0.2,
    ):

        super(GraphStateLSTM, self).__init__()

        self.encoder = encoder
        self.use_ner = use_ner

        self.linear_chem = nn.Linear(
            self.encoder.encoder_hidden_size
            + elmo_hidden_size
            + flair_hidden_size
            + self.encoder.word_embedding_dim
            + self.encoder.char_embedding_dim
            + self.encoder.pos_embedding_dim
            + (ner_hidden_size if self.use_ner else 0),
            glstm_hidden_size,
        )
        self.linear_dis = nn.Linear(
            self.encoder.encoder_hidden_size
            + elmo_hidden_size
            + flair_hidden_size
            + self.encoder.word_embedding_dim
            + self.encoder.char_embedding_dim
            + self.encoder.pos_embedding_dim
            + (ner_hidden_size if self.use_ner else 0),
            glstm_hidden_size,
        )

        self.linear_score = nn.Linear(2 * glstm_hidden_size + distant_embedding_dim, relation_classes)
        self.distant_embedding = nn.Embedding(max_distant, distant_embedding_dim)
        self.drop_out = nn.Dropout(drop_out)

        self.linear_ner_out = nn.Linear(2 * glstm_hidden_size, ner_classes)

        # self.ner_embedding = nn.Embedding(ner_classes, 50)
        # self.biaffine = BiaffineAttention(100, 100)
        self.ner_lstm = nn.LSTM(
            input_size=self.encoder.encoder_hidden_size
            + elmo_hidden_size
            + flair_hidden_size
            + self.encoder.word_embedding_dim
            + self.encoder.char_embedding_dim
            + self.encoder.pos_embedding_dim,
            hidden_size=ner_hidden_size,
            bidirectional=True,
            num_layers=lstm_layers,
        )

    def collect_entity_by_indices(self, representations, positions):

        batch, max_mentions, max_entity_span = positions.shape
        feature_dim = representations.shape[-1]

        positions = positions.view(batch, max_mentions * max_entity_span)
        # positions = [batch, max_mentions * max_entity_span ]

        collected_tensor = torch.gather(
            representations, 1, positions[..., None].expand(*positions.shape, representations.shape[-1])
        )

        collected_tensor = collected_tensor.view(batch, max_mentions, max_entity_span, feature_dim)

        return collected_tensor

    def cal_span_representation(self, collected_hidden_states, W_spans, ffn_alpha, mask=None):

        scores = torch.tanh(ffn_alpha(collected_hidden_states))

        # scores = [b, max_seq_length, max_spans, hidden_size]

        scores = torch.matmul(W_spans, scores.permute(0, 1, 3, 2))

        # scores = [b, max_seq_length, 4, max_spans]

        # print(scores.shape)
        # print(mask.shape)

        scores = scores.masked_fill(mask.unsqueeze(-2) == 0, -1e10)

        scores = torch.softmax(scores, dim=-1)

        # scores = [b, max_seq_length, 4, max_spans]

        span_representation = torch.matmul(collected_hidden_states.permute(0, 1, 3, 2), scores.permute(0, 1, 3, 2))

        # span_representation = [b, max_seq_length, hidden_size, 4]

        span_representation = torch.sum(span_representation, dim=-1)

        return span_representation

    def forward(self, inputs):

        (
            token_ids_tensor,
            token_ids_mask_tensor,
            pos_ids_tensor,
            char_ids_tensor,
            hypernym_ids_tensor,
            synonym_ids_tensor,
            elmo_tensor,
            flair_tensor,
            in_nodes_idx_tensor,
            in_nodes_mask_tensor,
            out_nodes_idx_tensor,
            out_nodes_mask_tensor,
            in_edge_idx_tensor,
            in_edge_idx_mask,
            out_edge_idx_tensor,
            out_edge_idx_mask,
            chem_entity_map_tensor,
            chem_entity_map_mask_tensor,
            dis_entity_map_tensor,
            dis_entity_map_mask_tensor,
            distant,
        ) = inputs

        node_cell, node_hidden, word_embedded = self.encoder(
            [
                token_ids_tensor,
                token_ids_mask_tensor,
                pos_ids_tensor,
                char_ids_tensor,
                hypernym_ids_tensor,
                synonym_ids_tensor,
                in_nodes_idx_tensor,
                in_nodes_mask_tensor,
                out_nodes_idx_tensor,
                out_nodes_mask_tensor,
                in_edge_idx_tensor,
                in_edge_idx_mask,
                out_edge_idx_tensor,
                out_edge_idx_mask,
            ]
        )

        entity_spans = chem_entity_map_tensor.shape[2]

        # node_cell = [batch_size, max_seq_length, hidden_size]
        # node_hidden = [batch_size, max_seq_length, hidden_size]

        # word_embedded = [batch_size, max_seq_length, hidden_size]

        # chem_entity_map_tensor = [batch_size, max_mention, max_entity_span]
        # chem_entity_map_mask_tensor = [batch_size, max_mention, max_entity_span]

        representations = self.drop_out(torch.cat([node_hidden, elmo_tensor, flair_tensor, word_embedded], dim=-1))

        # ner_hiddens, (h_n, c_n) = self.ner_lstm(representations.permute(1,0,2))

        # ner_logits = self.linear_ner_out(ner_hiddens.permute(1,0,2))
        # # ner_label_embedidng = self.ner_embedding(ner_logits.argmax(-1))

        if self.use_ner:
            ner_hiddens, (h_n, c_n) = self.ner_lstm(representations.permute(1, 0, 2))
            ner_logits = self.linear_ner_out(ner_hiddens.permute(1, 0, 2))
            representations = torch.cat([representations, ner_hiddens.permute(1, 0, 2)], dim=-1)

        collected_chem_entities = self.collect_entity_by_indices(representations, chem_entity_map_tensor)
        collected_chem_entities = collected_chem_entities * chem_entity_map_mask_tensor.unsqueeze(-1)
        chem_entities = torch.sum(collected_chem_entities, dim=2)

        chem_entities = torch.tanh(self.linear_chem(chem_entities))

        collected_dis_entities = self.collect_entity_by_indices(representations, dis_entity_map_tensor)
        collected_dis_entities = collected_dis_entities * dis_entity_map_mask_tensor.unsqueeze(-1)
        dis_entities = torch.sum(collected_dis_entities, dim=2)

        dis_entities = torch.tanh(self.linear_dis(dis_entities))

        distant_embedded = self.distant_embedding(distant)

        # chem_entities = [batch_size, max_mentions, feature_dim]
        # dis_entities = [batch_size, max_mentions, feature_dim]

        # tmp = torch.max(representations, dim =1)[0]

        bs, max_mentions, feature_dim = chem_entities.shape

        chem_entities_indices_mask = torch.sum(chem_entity_map_mask_tensor, dim=-1).type(torch.cuda.LongTensor)
        dis_entities_indices_mask = torch.sum(dis_entity_map_mask_tensor, dim=-1).type(torch.cuda.LongTensor)

        # print(chem_entities_indices_mask)

        # chem_entities_indices_mask = [batch_size, max_mentions]
        # chem_entities_indices_mask = [batch_size, max_mentions]

        concat_entities = torch.cat(
            [
                chem_entities.unsqueeze(2).repeat(1, 1, max_mentions, 1),
                dis_entities.unsqueeze(1).repeat(1, max_mentions, 1, 1),
            ],
            dim=-1,
        )
        # concat_entities = [batch_size, max_mentions, max_mentions, feature_dim * 2]

        # temp_chem = chem_entities.unsqueeze(2).repeat(1,1,max_mentions,1)
        # temp_dis = dis_entities.unsqueeze(1).repeat(1,max_mentions,1,1)
        # concat_entities = self.biaffine(temp_chem, temp_dis )

        # print(concat_entities.shape)
        concat_entities = torch.cat([concat_entities, distant_embedded], dim=-1)
        concat_entities = concat_entities.view(bs, max_mentions * max_mentions, 2 * feature_dim + 50)
        # concat_entities = [batch_size, max_mentions * max_mentions, feature_dim * 2]

        # concat_entities = torch.cat([concat_entities, distant_embedded],dim=-1)

        concat_indices_mask = chem_entities_indices_mask.unsqueeze(2).repeat(
            1, 1, max_mentions
        ) * dis_entities_indices_mask.unsqueeze(1).repeat(1, max_mentions, 1)
        # concat_indices_mask = [bs, max_mentions, max_mentions]

        concat_indices_mask = concat_indices_mask.view(bs, max_mentions * max_mentions)
        # concat_indices_mask = [bs, max_mentions * max_mentions]

        score = self.linear_score(self.drop_out(concat_entities))
        # score = [bs, max_mentions * max_mentions, 2]

        score = score.masked_fill(concat_indices_mask.unsqueeze(-1) == 0, -1e10)

        final_score = torch.max(score, dim=1)[0]

        if self.use_ner:
            return ner_logits, final_score

        # final_score = torch.logsumexp(score, dim =1)
        return final_score
