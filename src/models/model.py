import torch
from all_packages import *
from pytorch_transformers import BertModel
from sklearn import metrics
from src.models.layers import *
from src.utils import torch_utils
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR

from .utils import Accuracy


class LSR(nn.Module):
    def __init__(self, args):
        super().__init__()

        ## Init some commonly-used local vars
        data_word_vec = np.load(os.path.join(args.data_path, "vec.npy"))
        entity_type_size = args.pos_dim
        coref_size = args.coref_dim
        hidden_size = args.hidden_dim
        dropout_rate = args.dropout_rate
        dropout_emb = args.dropout_emb
        dropout_gcn = args.dropout_gcn

        max_length = 512
        relation_num = 97
        dis_size = 20

        ## Init class attributes
        self.use_struct_att = args.use_struct_att
        self.reasoner_layer_first = args.reasoner_layer_first
        self.reasoner_layer_second = args.reasoner_layer_second
        self.use_reasoning_block = args.use_reasoning_block
        self.relation_num = relation_num
        self.rel2id = json.load(open(os.path.join(args.data_path, "rel2id.json")))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        ## Init layers

        ## Load GloVe Embedding
        self.word_emb = nn.Embedding(data_word_vec.shape[0], data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(data_word_vec))
        if not args.finetune_emb:
            self.word_emb.weight.requires_grad = False

        self.ner_emb = nn.Embedding(13, entity_type_size, padding_idx=0)
        self.coref_embed = nn.Embedding(max_length, coref_size, padding_idx=0)

        input_size = data_word_vec.shape[1] + coref_size + entity_type_size  # + char_hidden

        self.linear_re = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_sent = nn.Linear(hidden_size * 2, hidden_size)
        self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)

        self.self_att = SelfAttention(hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size + dis_size, hidden_size + dis_size, hidden_size)
        self.dis_embed = nn.Embedding(20, dis_size, padding_idx=10)

        self.linear_output = nn.Linear(2 * hidden_size, relation_num)

        self.relu = nn.ReLU()

        self.dropout_rate = nn.Dropout(dropout_rate)

        self.rnn_sent = Encoder(input_size, hidden_size, dropout_emb, dropout_rate)
        self.hidden_size = hidden_size

        if self.use_struct_att == True:
            self.structInduction = StructInduction(hidden_size // 2, hidden_size, True)

        self.dropout_gcn = nn.Dropout(dropout_gcn)

        if self.use_reasoning_block:
            self.reasoner = nn.ModuleList()
            self.reasoner.append(
                DynamicReasoner(hidden_size, self.reasoner_layer_first, self.dropout_gcn)
            )
            self.reasoner.append(
                DynamicReasoner(hidden_size, self.reasoner_layer_second, self.dropout_gcn)
            )

    def doc_encoder(self, input_sent, context_seg):
        """
        :param sent: sent emb
        :param context_seg: segmentation mask for sentences in a document
        :return:
        """
        batch_size = context_seg.shape[0]
        docs_emb = []  # sentence embedding
        docs_len = []
        sents_emb = []

        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = (
                ((context_seg[batch_no] == 1).nonzero()).squeeze(-1).tolist()
            )  # array of start point for sentences in a document
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index : index + 1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index + 1 : index + 1])
                        sent_lens.append(index - pre_index)
                pre_index = index

            sents = pad_sequence(sent_list).permute(1, 0, 2)
            sent_lens_t = torch.LongTensor(sent_lens).cuda()
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(
                sents, sent_lens_t
            )  # sentence embeddings for a document.

            doc_emb = None
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim=0)

            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))

        docs_emb = pad_sequence(docs_emb).permute(1, 0, 2)  # B * # sentence * Dimention
        sents_emb = pad_sequence(sents_emb).permute(1, 0, 2)

        return docs_emb, sents_emb

    def forward(
        self,
        context_idxs,
        pos,
        context_ner,
        h_mapping,
        t_mapping,
        relation_mask,
        dis_h_2_t,
        dis_t_2_h,
        context_seg,
        mention_node_position,
        entity_position,
        mention_node_sent_num,
        all_node_num,
        entity_num_list,
        sdp_pos,
        sdp_num_list,
    ):
        """
        :param context_idxs: Token IDs
        :param pos: coref pos IDs
        :param context_ner: NER tag IDs
        :param h_mapping: Head
        :param t_mapping: Tail
        :param relation_mask: There are multiple relations for each instance so we need a mask in a batch
        :param dis_h_2_t: distance for head
        :param dis_t_2_h: distance for tail
        :param context_seg: mask for different sentences in a document
        :param mention_node_position: Mention node position
        :param entity_position: Entity node position
        :param mention_node_sent_num: number of mention nodes in each sentences of a document
        :param all_node_num: the number of nodes  (mention, entity, MDP) in a document
        :param entity_num_list: the number of entity nodes in each document
        :param sdp_pos: MDP node position
        :param sdp_num_list: the number of MDP node in each document
        :return:
        """

        # ==========STEP1: Encode the document============
        sent_emb = torch.cat(
            [self.word_emb(context_idxs), self.coref_embed(pos), self.ner_emb(context_ner)], dim=-1
        )
        docs_rep, sents_rep = self.doc_encoder(sent_emb, context_seg)

        max_doc_len = docs_rep.shape[1]
        context_output = self.dropout_rate(torch.relu(self.linear_re(docs_rep)))

        # ==========STEP2: Extract all node reps of a document graph============
        # extract Mention node representations"""
        mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()
        max_mention_num = max(mention_num_list)
        mentions_rep = torch.bmm(
            mention_node_position[:, :max_mention_num, :max_doc_len], context_output
        )  # mentions rep
        # extract MDP(meta dependency paths) node representations"""
        sdp_num_list = sdp_num_list.long().tolist()
        max_sdp_num = max(sdp_num_list)
        sdp_rep = torch.bmm(sdp_pos[:, :max_sdp_num, :max_doc_len], context_output)
        # extract Entity node representations"""
        entity_rep = torch.bmm(entity_position[:, :, :max_doc_len], context_output)
        """concatenate all nodes of an instance"""
        gcn_inputs = []
        all_node_num_batch = []
        for batch_no, (m_n, e_n, s_n) in enumerate(
            zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)
        ):
            m_rep = mentions_rep[batch_no][:m_n]
            e_rep = entity_rep[batch_no][:e_n]
            s_rep = sdp_rep[batch_no][:s_n]
            gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep), dim=0))
            node_num = m_n + e_n + s_n
            all_node_num_batch.append(node_num)

        gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
        output = gcn_inputs

        # ==========STEP3: Induce the Latent Structure============
        if self.use_reasoning_block:
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)

        elif self.use_struct_att:
            gcn_inputs, _ = self.structInduction(gcn_inputs)
            max_all_node_num = torch.max(all_node_num).item()
            assert gcn_inputs.shape[1] == max_all_node_num

        mention_node_position = mention_node_position.permute(0, 2, 1)
        output = torch.bmm(
            mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num]
        )
        context_output = torch.add(context_output, output)

        start_re_output = torch.matmul(
            h_mapping[:, :, :max_doc_len], context_output
        )  # aggregation
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output)  # aggregation

        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

        re_rep = self.dropout_rate(self.relu(self.bili(s_rep, t_rep)))

        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        return self.linear_output(re_rep)


class LSR_Bert(nn.Module):
    def __init__(self, args):
        super().__init__()

        entity_type_size = args.pos_dim
        coref_size = args.coref_dim
        hidden_size = args.hidden_dim
        dropout_rate = args.dropout_rate
        dropout_emb = args.dropout_emb
        dropout_gcn = args.dropout_gcn

        max_length = 512
        relation_num = 97
        dis_size = 20
        bert_hidden_size = 768

        ## Init class attributes
        self.use_struct_att = args.use_struct_att
        self.reasoner_layer_first = args.reasoner_layer_first
        self.reasoner_layer_second = args.reasoner_layer_second
        self.use_reasoning_block = args.use_reasoning_block
        self.relation_num = relation_num
        self.rel2id = json.load(open(os.path.join(args.data_path, "rel2id.json")))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        ## Init layers
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        print("loaded bert-base-uncased")

        self.linear_re = nn.Linear(bert_hidden_size, hidden_size)

        self.linear_sent = nn.Linear(hidden_size * 2, hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)

        self.self_att = SelfAttention(hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size + dis_size, hidden_size + dis_size, hidden_size)
        self.dis_embed = nn.Embedding(20, dis_size, padding_idx=10)

        self.linear_output = nn.Linear(2 * hidden_size, relation_num)

        self.relu = nn.ReLU()

        self.dropout_rate = nn.Dropout(dropout_rate)

        # self.rnn_sent = Encoder(input_size, hidden_size, dropout_emb, dropout_rate)
        self.hidden_size = hidden_size

        if self.use_struct_att == True:
            self.structInduction = StructInduction(hidden_size // 2, hidden_size, True)

        self.dropout_gcn = nn.Dropout(dropout_gcn)

        if self.use_reasoning_block:
            self.reasoner = nn.ModuleList()
            self.reasoner.append(
                DynamicReasoner(hidden_size, self.reasoner_layer_first, self.dropout_gcn)
            )
            self.reasoner.append(
                DynamicReasoner(hidden_size, self.reasoner_layer_second, self.dropout_gcn)
            )

    def doc_encoder(self, input_sent, context_seg):
        """
        :param sent: sent emb
        :param context_seg: segmentation mask for sentences in a document
        :return:
        """
        batch_size = context_seg.shape[0]
        docs_emb = []  # sentence embedding
        docs_len = []
        sents_emb = []

        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = (
                ((context_seg[batch_no] == 1).nonzero()).squeeze(-1).tolist()
            )  # array of start point for sentences in a document
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index : index + 1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index + 1 : index + 1])
                        sent_lens.append(index - pre_index)
                pre_index = index

            sents = pad_sequence(sent_list).permute(1, 0, 2)
            sent_lens_t = torch.LongTensor(sent_lens).cuda()
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(
                sents, sent_lens_t
            )  # sentence embeddings for a document.

            doc_emb = None
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim=0)

            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))

        docs_emb = pad_sequence(docs_emb).permute(1, 0, 2)  # B * # sentence * Dimention
        sents_emb = pad_sequence(sents_emb).permute(1, 0, 2)

        return docs_emb, sents_emb

    def forward(
        self,
        context_idxs,
        pos,
        context_ner,
        h_mapping,
        t_mapping,
        relation_mask,
        dis_h_2_t,
        dis_t_2_h,
        context_seg,
        mention_node_position,
        entity_position,
        mention_node_sent_num,
        all_node_num,
        entity_num_list,
        sdp_pos,
        sdp_num_list,
        context_masks,
        context_starts,
    ):
        """
        :param context_idxs: Token IDs
        :param pos: coref pos IDs
        :param context_ner: NER tag IDs
        :param h_mapping: Head
        :param t_mapping: Tail
        :param relation_mask: There are multiple relations for each instance so we need a mask in a batch
        :param dis_h_2_t: distance for head
        :param dis_t_2_h: distance for tail
        :param context_seg: mask for different sentences in a document
        :param mention_node_position: Mention node position
        :param entity_position: Entity node position
        :param mention_node_sent_num: number of mention nodes in each sentences of a document
        :param all_node_num: the number of nodes  (mention, entity, MDP) in a document
        :param entity_num_list: the number of entity nodes in each document
        :param sdp_pos: MDP node position
        :param sdp_num_list: the number of MDP node in each document
        :return:
        """

        # ==========STEP1: Encode the document============
        context_output = self.bert(context_idxs, attention_mask=context_masks)[0]
        context_output = [
            layer[starts.nonzero().squeeze(1)]
            for layer, starts in zip(context_output, context_starts)
        ]
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)
        context_output = torch.nn.functional.pad(
            context_output, (0, 0, 0, context_idxs.size(-1) - context_output.size(-2))
        )
        context_output = self.dropout_rate(torch.relu(self.linear_re(context_output)))
        max_doc_len = 512

        # max_doc_len = docs_rep.shape[1]

        # ==========STEP2: Extract all node reps of a document graph============
        # extract Mention node representations"""
        mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()
        max_mention_num = max(mention_num_list)
        mentions_rep = torch.bmm(
            mention_node_position[:, :max_mention_num, :max_doc_len], context_output
        )  # mentions rep
        # extract MDP(meta dependency paths) node representations"""
        sdp_num_list = sdp_num_list.long().tolist()
        max_sdp_num = max(sdp_num_list)
        sdp_rep = torch.bmm(sdp_pos[:, :max_sdp_num, :max_doc_len], context_output)
        # extract Entity node representations"""
        entity_rep = torch.bmm(entity_position[:, :, :max_doc_len], context_output)
        """concatenate all nodes of an instance"""
        gcn_inputs = []
        all_node_num_batch = []
        for batch_no, (m_n, e_n, s_n) in enumerate(
            zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)
        ):
            m_rep = mentions_rep[batch_no][:m_n]
            e_rep = entity_rep[batch_no][:e_n]
            s_rep = sdp_rep[batch_no][:s_n]
            gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep), dim=0))
            node_num = m_n + e_n + s_n
            all_node_num_batch.append(node_num)

        gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
        output = gcn_inputs

        # ==========STEP3: Induce the Latent Structure============
        if self.use_reasoning_block:
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)

        elif self.use_struct_att:
            gcn_inputs, _ = self.structInduction(gcn_inputs)
            max_all_node_num = torch.max(all_node_num).item()
            assert gcn_inputs.shape[1] == max_all_node_num

        mention_node_position = mention_node_position.permute(0, 2, 1)
        output = torch.bmm(
            mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num]
        )
        context_output = torch.add(context_output, output)

        start_re_output = torch.matmul(
            h_mapping[:, :, :max_doc_len], context_output
        )  # aggregation
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output)  # aggregation

        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

        re_rep = self.dropout_rate(self.relu(self.bili(s_rep, t_rep)))

        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        return self.linear_output(re_rep)


####### LitModel ##########################################################################################


class LSRLitModel(LightningModule):
    def __init__(self, model, hps):
        super().__init__()

        self._hparams = hps
        self.model = model

        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.acc_NA = Accuracy()
        self.acc_not_NA = Accuracy()
        self.acc_total = Accuracy()

        self.automatic_optimization = False

        self.lr = hps["lr"]

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        batch = batch[0]

        context_idxs = batch["context_idxs"].to(device)
        context_pos = batch["context_pos"].to(device)
        h_mapping = batch["h_mapping"].to(device)
        t_mapping = batch["t_mapping"].to(device)
        relation_mask = batch["relation_mask"].to(device)
        context_ner = batch["context_ner"].to(device)
        node_position = batch["node_position"].to(device)
        entity_position = batch["entity_position"].to(device)
        node_sent_num = batch["node_sent_num"].to(device)
        all_node_num = batch["all_node_num"].to(device)
        sdp_pos = batch["sdp_position"].to(device)

        ht_pair_pos = batch["ht_pair_pos"]
        context_seg = batch["context_seg"]
        dis_h_2_t = ht_pair_pos + 10
        dis_t_2_h = -ht_pair_pos + 10
        dis_h_2_t = dis_h_2_t.to(device)
        dis_t_2_h = dis_t_2_h.to(device)

        sdp_num = torch.tensor(batch["sdp_num"], device=device)
        entity_num = torch.tensor(batch["entity_num"], device=device)

        if "relation_multi_label" in batch:
            relation_multi_label = batch["relation_multi_label"].to(device)
        else:
            relation_multi_label = None
        if "relation_label" in batch:
            relation_label = batch["relation_label"]
        else:
            relation_label = None
        if "pos_idx" in batch:
            pos_idx = batch["pos_idx"]
        else:
            pos_idx = None

        return {
            "context_idxs": context_idxs,
            "context_pos": context_pos,
            "h_mapping": h_mapping,
            "t_mapping": t_mapping,
            "context_seg": context_seg,
            "relation_label": relation_label,
            "relation_multi_label": relation_multi_label,
            "relation_mask": relation_mask,
            "input_lengths": batch["input_lengths"],
            "context_ner": context_ner,
            "context_char_idxs": batch["context_char_idxs"],
            "ht_pair_pos": ht_pair_pos,
            "node_position": node_position,
            "node_sent_num": node_sent_num,
            "entity_position": entity_position,
            "entity_num": entity_num,
            "all_node_num": all_node_num,
            "pos_idx": pos_idx,
            "titles": batch["titles"],
            "indexes": batch["indexes"],
            "L_vertex": batch["L_vertex"],
            "labels": batch["labels"],
            "sent_num": batch["sentence_num"],
            "sdp_pos": sdp_pos,
            "sdp_num": sdp_num,
            "dis_h_2_t": dis_h_2_t,
            "dis_t_2_h": dis_t_2_h
            # "vertexsets": batch["vertexSets"],
        }

    ####### TRAIN ##########################################################################################

    def on_train_start(self) -> None:
        self.dev_score_list = []
        self.f1 = 0
        self.dev_score_list.append(self.f1)

        self.best_auc = 0.0
        self.best_f1 = 0.0
        self.best_epoch = 0

    def training_step(self, batch, batch_idx):
        gc.collect()

        opt = self.optimizers()
        opt.zero_grad()

        predict_re = self.model(
            batch["context_idxs"],
            batch["context_pos"],
            batch["context_ner"],
            batch["h_mapping"],
            batch["t_mapping"],
            batch["relation_mask"],
            batch["dis_h_2_t"],
            batch["dis_t_2_h"],
            batch["context_seg"],
            batch["node_position"],
            batch["entity_position"],
            batch["node_sent_num"],
            batch["all_node_num"],
            batch["entity_num"],
            batch["sdp_pos"],
            batch["sdp_num"],
        )

        ## Calculate loss and accuracies
        loss = torch.sum(
            self.criterion(predict_re, batch["relation_multi_label"])
            * batch["relation_mask"].unsqueeze(2)
        ) / torch.sum(batch["relation_mask"])

        relation_label = batch["relation_label"].data.cpu().numpy()
        output = torch.argmax(predict_re, dim=-1)
        output = output.data.cpu().numpy()

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                label = relation_label[i][j]
                if label < 0:
                    break
                if label == 0:
                    self.acc_NA.add(output[i][j] == label)
                else:
                    self.acc_not_NA.add(output[i][j] == label)

                self.acc_total.add(output[i][j] == label)

        ## Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True)

        self.manual_backward(loss)
        opt.step()

    def on_train_epoch_end(self) -> None:
        scheduler = self.lr_schedulers()
        optimizer = self.optimizers()

        # and epoch < self.evaluate_epoch:# and epoch < self.evaluate_epoch:
        if self.current_epoch > self._hparams["decay_epoch"]:
            if self.optim == "sgd" and self.f1 < self.dev_score_list[-1]:
                self.lr *= self._hparams["lr_decay"]
                for param_group in optimizer.param_groups:
                    param_group["lr"] = self.lr

            # epoch < 30:# and f1 < dev_score_list[-1]:
            if self.optim == "adam" and optimizer.param_groups[0]["lr"] > 1e-4:
                scheduler.step()

    ####### VALIDATION ##########################################################################################
    def on_validation_start(self) -> None:
        self.test_result = []
        self.test_result_ignore = []

        self.total_recall = 0
        self.total_recall_ignore = 0

        self.top1_acc = 0
        self.have_label = 0

    def validation_step(self, batch, batch_idx):

        predict_re = self.model(
            batch["context_idxs"],
            batch["context_pos"],
            batch["context_ner"],
            batch["h_mapping"],
            batch["t_mapping"],
            batch["relation_mask"],
            batch["dis_h_2_t"],
            batch["dis_t_2_h"],
            batch["context_seg"],
            batch["node_position"],
            batch["entity_position"],
            batch["node_sent_num"],
            batch["all_node_num"],
            batch["entity_num"],
            batch["sdp_pos"],
            batch["sdp_num"],
        )

        predict_re = torch.sigmoid(predict_re)
        predict_re = predict_re.data.cpu().numpy()

        labels = batch["labels"]
        indexes = batch["indexes"]
        titles = batch["titles"]
        for i, _ in enumerate(labels):
            label = labels[i]
            index = indexes[i]

            self.total_recall += len(label)
            for l in label.values():
                if not l:
                    self.total_recall_ignore += 1

            L = batch["L_vertex"][i]  # the number of entities in each instance.
            j = 0

            for h_idx in range(L):
                for t_idx in range(L):
                    if h_idx != t_idx:

                        pre_r = np.argmax(predict_re[i, j])
                        if (h_idx, t_idx, pre_r) in label:
                            self.top1_acc += 1

                        flag = False

                        for r in range(1, self.model.relation_num):
                            intrain = False

                            if (h_idx, t_idx, r) in label:
                                flag = True
                                if label[(h_idx, t_idx, r)] == True:
                                    intrain = True
                            if not intrain:
                                self.test_result_ignore.append(
                                    (
                                        (h_idx, t_idx, r) in label,
                                        float(predict_re[i, j, r]),
                                        titles[i],
                                        self.model.id2rel[r],
                                        index,
                                        h_idx,
                                        t_idx,
                                        r,
                                        np.argmax(predict_re[i, j]),
                                    )
                                )

                            self.test_result.append(
                                (
                                    (h_idx, t_idx, r) in label,
                                    float(predict_re[i, j, r]),
                                    titles[i],
                                    self.model.id2rel[r],
                                    index,
                                    h_idx,
                                    t_idx,
                                    r,
                                    pre_r,
                                )
                            )
                        if flag:
                            self.have_label += 1

                        j += 1

    def calc_val_metrics(self):
        input_theta = self._hparams["input_theta"]

        self.test_result_ignore.sort(key=lambda x: x[1], reverse=True)
        self.test_result.sort(key=lambda x: x[1], reverse=True)

        pr_x = []
        pr_y = []
        correct = 0
        w = 0

        if self.total_recall == 0:
            self.total_recall = 1  # for test

        for i, item in enumerate(self.test_result):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))  # precision
            pr_x.append(float(correct) / self.total_recall)  # recall
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype="float32")
        pr_y = np.asarray(pr_y, dtype="float32")
        f1_arr = 2 * pr_x * pr_y / (pr_x + pr_y + 1e-20)
        f1 = f1_arr.max()
        f1_pos = f1_arr.argmax()
        theta = self.test_result[f1_pos][1]

        if input_theta == -1:
            w = f1_pos

        auc = metrics.auc(x=pr_x, y=pr_y)

        input_theta = theta
        pr_x = []
        pr_y = []
        correct = 0
        w = 0
        for i, item in enumerate(self.test_result_ignore):
            correct += item[0]
            pr_y.append(float(correct) / (i + 1))
            pr_x.append(float(correct) / self.total_recall)
            if item[1] > input_theta:
                w = i

        pr_x = np.asarray(pr_x, dtype="float32")
        pr_y = np.asarray(pr_y, dtype="float32")
        f1_arr = 2 * pr_x * pr_y / (pr_x + pr_y + 1e-20)
        f1_ig = f1_arr.max()

        auc = metrics.auc(x=pr_x, y=pr_y)

        return f1, f1_ig, auc, pr_x, pr_y

    def on_validation_epoch_end(self):
        f1, f1_ig, auc, pr_x, pr_y = self.calc_val_metrics()

        self.log("val_f1", f1, on_step=False, on_epoch=True)
        self.log("val_f1_ign", f1_ig, on_step=False, on_epoch=True)
        self.log("val_auc", auc, on_step=False, on_epoch=True)

    def configure_optimizers(self):

        optimizer = torch_utils.get_optimizer(
            self._hparams["optim"], self.parameters(), self._hparams["lr"]
        )

        scheduler = ExponentialLR(optimizer, self._hparams["lr_decay"])

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_gradient_clipping(
        self, optimizer, optimizer_idx, gradient_clip_val, gradient_clip_algorithm
    ):
        if optimizer_idx == 0:
            # Lightning will handle the gradient clipping
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self._hparams["gradient_clip_val"],
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
