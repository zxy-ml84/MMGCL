import torch
from torch import nn
import numpy as np
import scipy.sparse as sp
import random
import torch.nn.functional as F

from torch_scatter import scatter
from base.abstract_recommender import GeneralRecommender


class MMGCL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGCL, self).__init__(config, dataset)
        self.config = config
        self.item_embeddings = None
        self.user_embeddings = None
        self.latent_dim = self.config['latent_dim']
        self.n_layers = self.config['layer_num']
        self.num_users = self.n_users
        self.num_items = self.n_items
        self.initializer = nn.init.xavier_uniform_
        self.sigmoid = nn.Sigmoid()

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.initializer(self.embedding_user.weight)
        self.initializer(self.embedding_item.weight)
        self.ui_interaction = dataset.inter_matrix(form='csr').astype(np.float32)

        self.__init_mm_feature(dataset)

    def __init_mm_feature(self, dataset):
        self.v_feat = F.normalize(self.v_feat, dim=1)
        self.v_dense = nn.Linear(self.v_feat.shape[1], self.latent_dim)
        self.initializer(self.v_dense.weight)
        if self.config["dataset"] == "amazon" or "tiktok":
            self.t_feat = F.normalize(self.t_feat, dim=1)
            self.t_dense = nn.Linear(self.t_feat.shape[1], self.latent_dim)
            self.initializer(self.t_dense.weight)
        if self.config["dataset"] == "tiktok":
            self.a_feat = F.normalize(self.a_feat, dim=1)
            self.a_dense = nn.Linear(self.a_feat.shape[1], self.latent_dim)
            self.initializer(self.a_dense.weight)
        else:
            self.a_feat = None
        self.item_feat_dim = self.latent_dim * 3
        self.read_user = nn.Linear(self.item_feat_dim, self.latent_dim)
        self.read_item = nn.Linear(self.item_feat_dim, self.latent_dim)
        self.initializer(self.read_user.weight)
        self.initializer(self.read_item.weight)
        sp_adj = self.convert_to_laplacian_mat(self.ui_interaction)
        self.norm_adj = self.convert_sparse_mat_to_tensor(sp_adj).to(self.device)

        if self.config["ssl_task"] == "ED+MM+CN":
            self.ssl_temp = self.config["ssl_temp"]
            self.dropout_rate = self.config["dropout_rate"]
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.ssl_criterion = nn.CrossEntropyLoss()
            self.p_vat = self.config['mask_p']

        elif self.config["ssl_task"] in ["ED+MM", "ED", "MM"]:
            self.ssl_criterion = nn.CrossEntropyLoss()
            self.ssl_temp = self.config["ssl_temp"]
            self.dropout_rate = self.config["dropout_rate"]
            self.dropout = nn.Dropout(p=self.dropout_rate)
            self.p_vat = self.config['mask_p']

    def sgl_encoder(self, user_emb, item_emb, perturbed_adj=None):
        ego_embeddings = torch.cat([user_emb, item_emb], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.n_layers):
            if perturbed_adj is not None:
                if isinstance(perturbed_adj, list):
                    ego_embeddings = torch.sparse.mm(perturbed_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(perturbed_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items])
        return user_all_embeddings, item_all_embeddings

    def compute(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight

        if self.v_feat is not None:
            self.v_dense_emb = self.v_dense(self.v_feat)  # v=>id
        self.i_emb_u, self.i_emb_i = self.sgl_encoder(users_emb, items_emb)
        self.v_emb_u, self.v_emb_i = self.sgl_encoder(users_emb, self.v_dense_emb)

        if self.config["dataset"] == "kwai":
            user = self.read_user(torch.cat([self.i_emb_u, self.v_emb_u],dim=1))
            item = self.read_user(torch.cat([self.i_emb_u, self.v_emb_i],dim=1))
        elif self.config["dataset"] == "amazon":
            if self.t_feat is not None:
                self.t_dense_emb = self.t_dense(self.t_feat)  # t=>id
                self.t_emb_u, self.t_emb_i = self.sgl_encoder(users_emb, self.t_dense_emb)
            user = self.read_user(torch.cat([self.i_emb_u, self.v_emb_u, self.t_emb_u], dim=1))
            item = self.read_item(torch.cat([self.i_emb_i, self.v_emb_i, self.t_emb_i], dim=1))
        elif self.config["dataset"] == "tiktok":
            if self.a_feat is not None:
                self.a_dense_emb = self.a_dense(self.a_feat)  # a=>id
                self.a_emb_u, self.a_emb_i = self.sgl_encoder(users_emb, self.a_dense_emb)
            if self.t_feat is not None:
                self.t_dense_emb = self.t_dense(self.t_feat)  # t=>id
                self.t_emb_u, self.t_emb_i = self.sgl_encoder(users_emb, self.t_dense_emb)
            user = self.read_user(torch.cat([self.i_emb_u, self.v_emb_u, self.a_emb_u, self.t_emb_u], dim=1))
            item = self.read_item(torch.cat([self.i_emb_i, self.v_emb_i, self.a_emb_u, self.t_emb_i], dim=1))

        return user, item

    def random_graph_augment(self, aug_type):
        dropped_mat = None
        if aug_type == 1:
            dropped_mat = self.node_dropout(self.ui_interaction, self.dropout_rate)
        elif aug_type == 0:
            dropped_mat = self.edge_dropout(self.ui_interaction, self.dropout_rate)
        dropped_mat = self.convert_to_laplacian_mat(dropped_mat)
        return self.convert_sparse_mat_to_tensor(dropped_mat).cuda()

    def graph_reconstruction(self, aug_type):
        if aug_type == 0:
            dropped_adj = self.random_graph_augment(aug_type)
        elif aug_type == 1:
            dropped_adj = self.random_graph_augment(aug_type)
        else:
            dropped_adj = []
            for k in range(self.n_layers):
                dropped_adj.append(self.random_graph_augment())
        return dropped_adj

    def modality_edge_dropout_emb(self, u_ids, pos_ids, neg_ids):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        v_dense = self.v_dense_emb
        if self.config["dataset"] == "kwai":
            v_dense = self.v_dense_emb
        elif self.config["dataset"] == "amazon":
            v_dense = self.v_dense_emb
            t_dense = self.t_dense_emb
        else:
            v_dense = self.v_dense_emb
            a_dense = self.a_dense_emb
            t_dense = self.t_dense_emb

        perturbed_adj = self.graph_reconstruction(aug_type=0)
        i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb, perturbed_adj)
        v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)

        i_emb_u_sub, i_emb_i_sub, i_emb_neg_i_sub = i_emb_u_sub[u_ids], i_emb_i_sub[pos_ids], i_emb_i_sub[neg_ids]
        v_emb_u_sub, v_emb_i_sub, v_emb_neg_i_sub = v_emb_u_sub[u_ids], v_emb_i_sub[pos_ids], v_emb_i_sub[neg_ids]

        if self.config["dataset"] == "kwai":
            users_sub = self.read_user(torch.cat([i_emb_u_sub, v_emb_u_sub], dim=1))
            items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub], dim=1))
            neg_items_sub = self.embedding_item_after_GCN(torch.cat([i_emb_i_sub, v_emb_neg_i_sub], dim=1))

        elif self.config["dataset"] == "amazon":
            t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, t_dense, perturbed_adj)
            t_emb_u_sub, t_emb_i_sub, t_emb_neg_i_sub = t_emb_u_sub[u_ids], t_emb_i_sub[pos_ids], t_emb_i_sub[neg_ids]

            users_sub = self.read_user(torch.cat([i_emb_u_sub, v_emb_u_sub, t_emb_u_sub], dim=1))
            items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, t_emb_i_sub], dim=1))
            # choose certain modality to replace with negative emb
            neg_items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, t_emb_neg_i_sub], dim=1))

        elif self.config["dataset"] == "tiktok":
            a_emb_u_sub, a_emb_i_sub = self.sgl_encoder(users_emb, a_dense, perturbed_adj)
            a_emb_u_sub, a_emb_i_sub, a_emb_neg_i_sub = a_emb_u_sub[u_ids], a_emb_i_sub[pos_ids], a_emb_i_sub[neg_ids]

            t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, t_dense, perturbed_adj)
            t_emb_u_sub, t_emb_i_sub, t_emb_neg_i_sub = t_emb_u_sub[u_ids], t_emb_i_sub[pos_ids], t_emb_i_sub[neg_ids]

            users_sub = self.read_user(torch.cat([i_emb_u_sub, v_emb_u_sub, a_emb_u_sub, t_emb_u_sub], dim=1))
            items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, a_emb_i_sub, t_emb_i_sub], dim=1))
            # choose certain modality to replace with negative emb
            neg_items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, a_emb_i_sub, t_emb_neg_i_sub], dim=1))

        users_sub = F.normalize(users_sub, dim=1)
        items_sub = F.normalize(items_sub, dim=1)
        neg_items_sub = F.normalize(neg_items_sub, dim=1)

        return users_sub, items_sub, neg_items_sub

    def modality_masking_emb(self, u_ids, pos_ids, neg_ids, p_vat):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        v_dense = self.v_dense_emb
        if self.config["dataset"] == "kwai":
            v_dense = self.v_dense_emb
        elif self.config["dataset"] == "amazon":
            v_dense = self.v_dense_emb
            t_dense = self.t_dense_emb
        elif self.config["dataset"] == "tiktok":
            v_dense = self.v_dense_emb
            a_dense = self.a_dense_emb
            t_dense = self.t_dense_emb

        # p value for masking certain modality
        perturbed_adj = self.graph_reconstruction(aug_type=1)
        if self.config["dataset"] == "kwai":
            v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)
            i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb)

            i_emb_u_sub, i_emb_i_sub, i_emb_neg_i_sub = i_emb_u_sub[u_ids], i_emb_i_sub[pos_ids], i_emb_i_sub[neg_ids]
            v_emb_u_sub, v_emb_i_sub, v_emb_neg_i_sub = v_emb_u_sub[u_ids], v_emb_i_sub[pos_ids], v_emb_i_sub[neg_ids]

            users_sub = self.read_user(torch.cat([i_emb_u_sub, v_emb_u_sub], dim=1))
            items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub], dim=1))
            neg_items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_neg_i_sub], dim=1))

        elif self.config["dataset"] == "amazon":
            modalities = ["image", "text"]
            modality_index = np.random.choice(len(modalities), p=p_vat)
            modality = modalities[modality_index]
            if modality == "image":
                v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)
                i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb)
                t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, t_dense)
            elif modality == "text":
                t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)
                i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb)
                v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, t_dense)

            i_emb_u_sub, i_emb_i_sub, i_emb_neg_i_sub = i_emb_u_sub[u_ids], i_emb_i_sub[pos_ids], i_emb_i_sub[neg_ids]
            v_emb_u_sub, v_emb_i_sub, v_emb_neg_i_sub = v_emb_u_sub[u_ids], v_emb_i_sub[pos_ids], v_emb_i_sub[neg_ids]
            t_emb_u_sub, t_emb_i_sub, t_emb_neg_i_sub = t_emb_u_sub[u_ids], t_emb_i_sub[pos_ids], t_emb_i_sub[neg_ids]

            users_sub = self.read_user(torch.cat([i_emb_u_sub, v_emb_u_sub, t_emb_u_sub], dim=1))
            items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, t_emb_i_sub], dim=1))
            # choose certain modality to replace with negative emb
            neg_items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_neg_i_sub, t_emb_neg_i_sub], dim=1))

        elif self.config["dataset"] == "tiktok":
            modalities = ["image", "text", "audio"]
            modality_index = np.random.choice(len(modalities), p=p_vat)
            modality = modalities[modality_index]
            if modality == "image":
                v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)
                i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb)
                a_emb_u_sub, a_emb_i_sub = self.sgl_encoder(users_emb, a_dense)
                t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, t_dense)
            elif modality == "audio":
                a_emb_u_sub, a_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)
                i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb)
                v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, a_dense)
                t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, t_dense)
            else:
                t_emb_u_sub, t_emb_i_sub = self.sgl_encoder(users_emb, v_dense, perturbed_adj)
                i_emb_u_sub, i_emb_i_sub = self.sgl_encoder(users_emb, items_emb)
                a_emb_u_sub, a_emb_i_sub = self.sgl_encoder(users_emb, a_dense)
                v_emb_u_sub, v_emb_i_sub = self.sgl_encoder(users_emb, t_dense)

            i_emb_u_sub, i_emb_i_sub, i_emb_neg_i_sub = i_emb_u_sub[u_ids], i_emb_i_sub[pos_ids], i_emb_i_sub[neg_ids]
            v_emb_u_sub, v_emb_i_sub, v_emb_neg_i_sub = v_emb_u_sub[u_ids], v_emb_i_sub[pos_ids], v_emb_i_sub[neg_ids]
            a_emb_u_sub, a_emb_i_sub, a_emb_neg_i_sub = a_emb_u_sub[u_ids], a_emb_i_sub[pos_ids], a_emb_i_sub[neg_ids]
            t_emb_u_sub, t_emb_i_sub, t_emb_neg_i_sub = t_emb_u_sub[u_ids], t_emb_i_sub[pos_ids], t_emb_i_sub[neg_ids]

            users_sub = self.read_user(torch.cat([i_emb_u_sub, v_emb_u_sub, a_emb_u_sub, t_emb_u_sub], dim=1))
            items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, a_emb_i_sub, t_emb_i_sub], dim=1))
            neg_items_sub = self.read_item(torch.cat([i_emb_i_sub, v_emb_i_sub, a_emb_i_sub, t_emb_neg_i_sub], dim=1))

        users_sub = F.normalize(users_sub, dim=1)
        items_sub = F.normalize(items_sub, dim=1)
        neg_items_sub = F.normalize(neg_items_sub, dim=1)

        return users_sub, items_sub, neg_items_sub

    def cal_multiview_MM_ED_CN(self, users, pos_items, neg_items):
        if self.config["ssl_task"] == "ED+MM":
            users_sub_1, items_sub_1, _ = self.modality_edge_dropout_emb(users, pos_items, neg_items)
            users_sub_2, items_sub_2, _ = self.modality_masking_emb(users, pos_items, neg_items, self.p_vat)

            logits_1 = torch.mm(users_sub_1, items_sub_1.T)
            logits_1 /= self.ssl_temp
            labels_1 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_1 = F.cross_entropy(logits_1, labels_1)

            logits_2 = torch.mm(users_sub_1, items_sub_1.T)
            logits_2 /= self.ssl_temp
            labels_2 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_2 = F.cross_entropy(logits_2, labels_2)
            ssl_loss = ssl_loss_1 + ssl_loss_2

            return ssl_loss

        elif self.config["ssl_task"] == "ED+MM+CN":
            users_sub_1, items_sub_1, neg_items_sub_1 = self.modality_edge_dropout_emb(users, pos_items, neg_items)
            users_sub_2, items_sub_2, neg_items_sub_2 = self.modality_masking_emb(users, pos_items, neg_items,
                                                                                  self.p_vat)

            logits_1 = torch.mm(users_sub_1, items_sub_1.T)
            logits_1 /= self.ssl_temp
            labels_1 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_1 = F.cross_entropy(logits_1, labels_1)

            logits_2 = torch.mm(users_sub_1, items_sub_1.T)
            logits_2 /= self.ssl_temp
            labels_2 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_2 = F.cross_entropy(logits_2, labels_2)

            logits_3 = torch.mm(users_sub_1, neg_items_sub_1.T)
            logits_3 /= self.ssl_temp
            labels_3 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_3 = - F.cross_entropy(logits_3, labels_3)
            ssl_loss = ssl_loss_1 + ssl_loss_2

            return ssl_loss

        elif self.config["ssl_task"] == "ED":
            users_sub_1, items_sub_1, _ = self.modality_edge_dropout_emb(users, pos_items, neg_items)
            users_sub_2, items_sub_2, _ = self.modality_edge_dropout_emb(users, pos_items, neg_items)

            logits_1 = torch.mm(users_sub_1, items_sub_1.T)
            logits_1 /= self.ssl_temp
            labels_1 = torch.tensor(list(range(users_sub_2.shape[0]))).to(self.device)
            ssl_loss_1 = F.cross_entropy(logits_1, labels_1)

            logits_2 = torch.mm(users_sub_2, items_sub_2.T)
            logits_2 /= self.ssl_temp
            labels_2 = torch.tensor(list(range(items_sub_2.shape[0]))).to(self.device)
            ssl_loss_2 = F.cross_entropy(logits_2, labels_2)
            ssl_loss = ssl_loss_1 + ssl_loss_2

            return ssl_loss

    def extract_ui_embeddings(self, input_users, positive_items, negative_items=None):
        self.user_embeddings, self.item_embeddings = self.compute()
        user_emb = self.user_embeddings[input_users]
        positive_emb = self.item_embeddings[positive_items]
        negative_emb = None if negative_items is None else self.item_embeddings[negative_items]

        return user_emb, positive_emb, negative_emb

    def bpr_loss(self, input_users, positive_items, negative_items):
        users_emb, pos_emb, neg_emb = self.extract_ui_embeddings(input_users, positive_items, negative_items)
        pos_score = torch.mul(users_emb, pos_emb).sum(dim=1)
        neg_score = torch.mul(users_emb, neg_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)

        ###

    def calculate_loss(self, interaction):
        # multi-task loss
        users, pos, neg = interaction[0], interaction[1], interaction[2]
        main_loss = self.bpr_loss(users, pos, neg)
        ssl_loss = self.cal_multiview_MM_ED_CN(users, pos, neg)

        return main_loss + self.config['ssl_alpha'] * ssl_loss

    def forward(self, users, items):
        user_embeddings, item_embeddings = self.compute()
        users_emb = user_embeddings[users]
        items_emb = item_embeddings[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma.detach()

    def convert_to_laplacian_mat(self, adj_mat):
        adj_shape = adj_mat.get_shape()
        n_nodes = adj_shape[0] + adj_shape[1]
        (user_np_keep, item_np_keep) = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_shape[0])), shape=(n_nodes, n_nodes),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    def normalize_graph_mat(self, adj_mat):
        shape = adj_mat.get_shape()
        rowsum = np.array(adj_mat.sum(1))
        if shape[0] == shape[1]:
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            norm_adj_mat = norm_adj_tmp.dot(d_mat_inv)
        else:
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_mat = d_mat_inv.dot(adj_mat)
        return norm_adj_mat

    def convert_sparse_mat_to_tensor(self, mat):
        coo = mat.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def node_dropout(self, sp_adj, drop_rate):
        adj_shape = sp_adj.get_shape()
        row_idx, col_idx = sp_adj.nonzero()
        drop_user_idx = random.sample(range(adj_shape[0]), int(adj_shape[0] * drop_rate))
        drop_item_idx = random.sample(range(adj_shape[1]), int(adj_shape[1] * drop_rate))
        indicator_user = np.ones(adj_shape[0], dtype=np.float32)
        indicator_item = np.ones(adj_shape[1], dtype=np.float32)
        indicator_user[drop_user_idx] = 0.
        indicator_item[drop_item_idx] = 0.
        diag_indicator_user = sp.diags(indicator_user)
        diag_indicator_item = sp.diags(indicator_item)
        mat = sp.csr_matrix(
            (np.ones_like(row_idx, dtype=np.float32), (row_idx, col_idx)),
            shape=(adj_shape[0], adj_shape[1]))
        mat_prime = diag_indicator_user.dot(mat).dot(diag_indicator_item)
        return mat_prime

    def edge_dropout(self, sp_adj, drop_rate):
        adj_shape = sp_adj.get_shape()
        edge_count = sp_adj.count_nonzero()
        row_idx, col_idx = sp_adj.nonzero()
        keep_idx = random.sample(range(edge_count), int(edge_count * (1 - drop_rate)))
        user_np = np.array(row_idx)[keep_idx]
        item_np = np.array(col_idx)[keep_idx]
        edges = np.ones_like(user_np, dtype=np.float32)
        dropped_adj = sp.csr_matrix((edges, (user_np, item_np)), shape=adj_shape)
        return dropped_adj



