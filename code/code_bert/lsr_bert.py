import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from models.encoder import Encoder
from models.attention import SelfAttention
from models.reasoner import DynamicReasoner
from models.reasoner import StructInduction
from pytorch_transformers import BertModel

class LSR(nn.Module):
    def __init__(self, config):
        super(LSR, self).__init__()
        self.config = config


        self.bert = BertModel.from_pretrained('bert-base-uncased')
        print("loaded bert-base-uncased")

        hidden_size = config.rnn_hidden
        bert_hidden_size = 768
        self.linear_re = nn.Linear(bert_hidden_size, hidden_size)

        self.linear_sent = nn.Linear(hidden_size * 2,  hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size, hidden_size, hidden_size)

        self.self_att = SelfAttention(hidden_size)

        self.bili = torch.nn.Bilinear(hidden_size+config.dis_size,  hidden_size+config.dis_size, hidden_size)
        self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)

        self.linear_output = nn.Linear(2 * hidden_size, config.relation_num)

        self.relu = nn.ReLU()

        self.dropout_rate = nn.Dropout(config.dropout_rate)

        #self.rnn_sent = Encoder(input_size, hidden_size, config.dropout_emb, config.dropout_rate)
        self.hidden_size = hidden_size

        self.use_struct_att = config.use_struct_att
        if  self.use_struct_att == True:
            self.structInduction = StructInduction(hidden_size // 2, hidden_size, True)

        self.dropout_gcn = nn.Dropout(config.dropout_gcn)
        self.reasoner_layer_first = config.reasoner_layer_first
        self.reasoner_layer_second = config.reasoner_layer_second
        self.use_reasoning_block = config.use_reasoning_block
        if self.use_reasoning_block:
            self.reasoner = nn.ModuleList()
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_first, self.dropout_gcn))
            self.reasoner.append(DynamicReasoner(hidden_size, self.reasoner_layer_second, self.dropout_gcn))

        NUM_DEPENDENCY = 6
        dependency_embd = [
            nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.empty(1, 2 * hidden_size, requires_grad=False if i == 0 else True)
                )
            )
            for i in range(NUM_DEPENDENCY)
        ]
        self.dependency_embd = torch.cat(dependency_embd)

    def doc_encoder(self, input_sent, context_seg):
        """
        :param sent: sent emb
        :param context_seg: segmentation mask for sentences in a document
        :return:
        """
        batch_size = context_seg.shape[0]
        docs_emb = [] # sentence embedding
        docs_len = []
        sents_emb = []

        for batch_no in range(batch_size):
            sent_list = []
            sent_lens = []
            sent_index = ((context_seg[batch_no] == 1).nonzero()).squeeze(-1).tolist() # array of start point for sentences in a document
            pre_index = 0
            for i, index in enumerate(sent_index):
                if i != 0:
                    if i == 1:
                        sent_list.append(input_sent[batch_no][pre_index:index+1])
                        sent_lens.append(index - pre_index + 1)
                    else:
                        sent_list.append(input_sent[batch_no][pre_index+1:index+1])
                        sent_lens.append(index - pre_index)
                pre_index = index

            sents = pad_sequence(sent_list).permute(1,0,2)
            sent_lens_t = torch.LongTensor(sent_lens).cuda()
            docs_len.append(sent_lens)
            sents_output, sent_emb = self.rnn_sent(sents, sent_lens_t) # sentence embeddings for a document.

            doc_emb = None
            for i, (sen_len, emb) in enumerate(zip(sent_lens, sents_output)):
                if i == 0:
                    doc_emb = emb[:sen_len]
                else:
                    doc_emb = torch.cat([doc_emb, emb[:sen_len]], dim = 0)

            docs_emb.append(doc_emb)
            sents_emb.append(sent_emb.squeeze(1))

        docs_emb = pad_sequence(docs_emb).permute(1,0,2) # B * # sentence * Dimention
        sents_emb = pad_sequence(sents_emb).permute(1,0,2)

        return docs_emb, sents_emb

    def get_relation_embd(self, structure_mask):

        self.dependency_embd = self.dependency_embd.to(structure_mask.device)

        relations = F.one_hot(structure_mask, num_classes=6).float()
        # [bz, max_len, max_len, 6]

        relation_embd = torch.einsum("bxyn,nd->bxyd", relations, self.dependency_embd)
        # [bz, max_len, max_len, 2*hid_size]

        return relation_embd

    def forward(self, context_idxs, pos, context_ner, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, context_seg, mention_node_position, entity_position,
                mention_node_sent_num, all_node_num, entity_num_list, sdp_pos, sdp_num_list, context_masks,
                context_starts, structure_mask):
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

        '''===========STEP1: Encode the document============='''
        context_output = self.bert(context_idxs, attention_mask=context_masks)[0]
        context_output = [layer[starts.nonzero().squeeze(1)]
                   for layer, starts in zip(context_output, context_starts)]
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)
        context_output = torch.nn.functional.pad(context_output,(0,0,0,context_idxs.size(-1)-context_output.size(-2)))
        context_output = self.dropout_rate(torch.relu(self.linear_re(context_output)))
        max_doc_len = 512

        # max_doc_len = docs_rep.shape[1]

        ## NOTE: Enable this to use structure mask
        # ==========STEP1.1: Encode structure into context embedding============
        structure_mask = self.get_relation_embd(structure_mask)
        # [bz, max_len, max_len, 2*hid_size]

        context_output1 = context_output.unsqueeze(1).repeat(1, max_doc_len, 1, 1)
        context_output2 = context_output.unsqueeze(2).repeat(1, 1, max_doc_len, 1)
        context_output_ = torch.cat((context_output1, context_output2), -1)
        # [bz, max_len, max_len, 2*hid_size]

        context_output_ = context_output_.unsqueeze(3)
        structure_mask = structure_mask.unsqueeze(-1)

        structure_embd = context_output_ @ structure_mask
        # [bz, max_len, max_len, 1, 1]
        structure_embd = structure_embd.squeeze().mean(dim=-1)
        # [bz, max_len]

        context_output = context_output + structure_embd.unsqueeze(-1)
        # [bz, max_len, hid_dim]

        '''===========STEP2: Extract all node reps of a document graph============='''
        '''extract Mention node representations'''
        mention_num_list = torch.sum(mention_node_sent_num, dim=1).long().tolist()
        max_mention_num = max(mention_num_list)
        mentions_rep = torch.bmm(mention_node_position[:, :max_mention_num, :max_doc_len], context_output) # mentions rep
        '''extract MDP(meta dependency paths) node representations'''
        sdp_num_list = sdp_num_list.long().tolist()
        max_sdp_num = max(sdp_num_list)
        sdp_rep = torch.bmm(sdp_pos[:,:max_sdp_num, :max_doc_len], context_output)
        '''extract Entity node representations'''
        entity_rep = torch.bmm(entity_position[:,:,:max_doc_len], context_output)
        '''concatenate all nodes of an instance'''
        gcn_inputs = []
        all_node_num_batch = []
        for batch_no, (m_n, e_n, s_n) in enumerate(zip(mention_num_list, entity_num_list.long().tolist(), sdp_num_list)):
            m_rep = mentions_rep[batch_no][:m_n]
            e_rep = entity_rep[batch_no][:e_n]
            s_rep = sdp_rep[batch_no][:s_n]
            gcn_inputs.append(torch.cat((m_rep, e_rep, s_rep),dim=0))
            node_num = m_n + e_n + s_n
            all_node_num_batch.append(node_num)

        gcn_inputs = pad_sequence(gcn_inputs).permute(1, 0, 2)
        output = gcn_inputs

        '''===========STEP3: Induce the Latent Structure============='''
        if self.use_reasoning_block:
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)

        elif self.use_struct_att:
            gcn_inputs, _ = self.structInduction(gcn_inputs)
            max_all_node_num = torch.max(all_node_num).item()
            assert (gcn_inputs.shape[1] == max_all_node_num)

        mention_node_position = mention_node_position.permute(0, 2, 1)
        output = torch.bmm(mention_node_position[:, :max_doc_len, :max_mention_num], output[:, :max_mention_num])
        context_output = torch.add(context_output, output)

        start_re_output = torch.matmul(h_mapping[:, :, :max_doc_len], context_output) # aggregation
        end_re_output = torch.matmul(t_mapping[:, :, :max_doc_len], context_output) # aggregation

        s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
        t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

        re_rep = self.dropout_rate(self.relu(self.bili(s_rep, t_rep)))

        re_rep = self.self_att(re_rep, re_rep, relation_mask)
        return self.linear_output(re_rep)

