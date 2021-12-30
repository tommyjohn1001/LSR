from collections import defaultdict
from operator import add

from all_packages import *
from torch.utils.data import IterableDataset


class IterDataset(IterableDataset):
    def __init__(self, args, prefix) -> None:
        super().__init__()

        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.prefix = prefix

        self.rel2id, self.id2rel = None, None

        self.data_word, self.data_pos, self.data_ner, self.data_char = None, None, None, None

        self.data_node_position = None
        self.data_node_position_sent = None
        self.data_node_sent_num = None
        self.data_node_num = None

        self.n_batches = 0

        self.use_bag = False
        self.use_gpu = True
        self.is_training = True
        self.max_length = 512
        self.pos_num = 2 * self.max_length
        self.entity_num = self.max_length

        self.relation_num = 97
        self.ner_vocab_len = 13

        self.max_sent_len = 200
        self.max_entity_num = 100
        self.max_sent_num = 30
        self.max_node_num = 200
        self.max_node_per_sent = 40

        self.checkpoint_dir = "./checkpoint"
        self.fig_result_dir = "./fig_result"

        self.h_t_limit = 1800

        self.test_relation_limit = 1800
        self.char_limit = 16
        self.sent_limit = 25
        self.dis2idx = np.zeros((512), dtype="int64")
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

    def load_data(self):
        logger.info(f"Load dataset: {self.prefix}")

        self.rel2id = json.load(open(os.path.join(self.data_path, "rel2id.json")))
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.data_word = np.load(os.path.join(self.data_path, self.prefix + "_word.npy"))
        self.data_pos = np.load(os.path.join(self.data_path, self.prefix + "_pos.npy"))
        self.data_ner = np.load(os.path.join(self.data_path, self.prefix + "_ner.npy"))
        self.data_char = np.load(os.path.join(self.data_path, self.prefix + "_char.npy"))

        self.data_node_position = np.load(
            os.path.join(self.data_path, self.prefix + "_node_position.npy")
        )

        self.data_node_position_sent = np.load(
            os.path.join(self.data_path, self.prefix + "_node_position_sent.npy")
        )
        # self.data_adj = np.load(os.path.join(self.data_path, self.prefix+'_adj.npy'))

        self.data_node_sent_num = np.load(
            os.path.join(self.data_path, self.prefix + "_node_sent_num.npy")
        )

        self.data_node_num = np.load(os.path.join(self.data_path, self.prefix + "_node_num.npy"))
        self.data_entity_position = np.load(
            os.path.join(self.data_path, self.prefix + "_entity_position.npy")
        )
        self.file = json.load(open(os.path.join(self.data_path, self.prefix + ".json")))
        self.data_seg = np.load(os.path.join(self.data_path, self.prefix + "_seg.npy"))
        self.data_len = self.data_word.shape[0]

        assert self.data_len == len(self.file)

        self.data_sdp_position = np.load(
            os.path.join(self.data_path, self.prefix + "_sdp_position.npy")
        )
        self.data_sdp_num = np.load(os.path.join(self.data_path, self.prefix + "_sdp_num.npy"))

        self.n_batches = self.data_word.shape[0] // self.batch_size
        if self.data_word.shape[0] % self.batch_size != 0:
            self.n_batches += 1

        self.order = list(range(self.data_len))
        if self.prefix == TEST_PREFIX:
            self.order.sort(key=lambda x: np.sum(self.data_word[x] > 0), reverse=True)

        logger.info(f"Finish reading, no. read documents: {self.data_len}")

    def __iter__(self):
        context_idxs = torch.LongTensor(self.batch_size, self.max_length)
        context_pos = torch.LongTensor(self.batch_size, self.max_length)
        if self.prefix == TEST_PREFIX:
            h_mapping = torch.Tensor(self.batch_size, self.test_relation_limit, self.max_length)
            t_mapping = torch.Tensor(self.batch_size, self.test_relation_limit, self.max_length)
        else:
            h_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length)
            t_mapping = torch.Tensor(self.batch_size, self.h_t_limit, self.max_length)
        context_ner = torch.LongTensor(self.batch_size, self.max_length)
        context_char_idxs = torch.LongTensor(self.batch_size, self.max_length, self.char_limit)
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit)
        ht_pair_pos = torch.LongTensor(self.batch_size, self.h_t_limit)
        context_seg = torch.LongTensor(self.batch_size, self.max_length)

        node_position_sent = torch.zeros(
            self.batch_size, self.max_sent_num, self.max_node_per_sent, self.max_sent_len
        ).float()

        node_position = torch.zeros(self.batch_size, self.max_node_num, self.max_length).float()
        entity_position = torch.zeros(
            self.batch_size, self.max_entity_num, self.max_length
        ).float()
        node_num = torch.zeros(self.batch_size, 1).long()

        node_sent_num = torch.zeros(self.batch_size, self.max_sent_num).float()

        sdp_position = torch.zeros(self.batch_size, self.max_entity_num, self.max_length).float()
        sdp_num = torch.zeros(self.batch_size, 1).long()

        for b in range(self.n_batches):
            entity_num = []
            sentence_num = []
            sentence_len = []
            node_num_per_sent = []

            start_id = b * self.batch_size
            cur_bsz = min(self.batch_size, self.data_len - start_id)
            cur_batch = list(self.order[start_id : start_id + cur_bsz])

            for mapping in [h_mapping, t_mapping, relation_mask]:
                mapping.zero_()

            ht_pair_pos.zero_()

            max_h_t_cnt = 1

            cur_batch.sort(key=lambda x: np.sum(self.data_word[x] > 0), reverse=True)

            labels = []

            L_vertex = []
            titles = []
            indexes = []
            sdp_nums = []

            vertexSets = []

            for i, index in enumerate(cur_batch):
                context_idxs[i].copy_(torch.from_numpy(self.data_word[index, :]))
                context_pos[i].copy_(torch.from_numpy(self.data_pos[index, :]))
                context_char_idxs[i].copy_(torch.from_numpy(self.data_char[index, :]))
                context_ner[i].copy_(torch.from_numpy(self.data_ner[index, :]))
                context_seg[i].copy_(torch.from_numpy(self.data_seg[index, :]))

                idx2label = defaultdict(list)
                ins = self.file[index]

                for label in ins["labels"]:
                    idx2label[(label["h"], label["t"])].append(label["r"])

                node_position[i].copy_(torch.from_numpy(self.data_node_position[index]))
                node_position_sent[i].copy_(torch.from_numpy(self.data_node_position_sent[index]))

                node_sent_num[i].copy_(torch.from_numpy(self.data_node_sent_num[index]))

                node_num[i].copy_(torch.from_numpy(self.data_node_num[index]))
                entity_position[i].copy_(torch.from_numpy(self.data_entity_position[index]))
                entity_num.append(len(ins["vertexSet"]))
                sentence_num.append(len(ins["sents"]))
                sentence_len.append(
                    max([len(sent) for sent in ins["sents"]])
                )  # max sent len of a document
                node_num_per_sent.append(max(node_sent_num[i].tolist()))

                sdp_position[i].copy_(torch.from_numpy(self.data_sdp_position[index]))
                sdp_num[i].copy_(torch.from_numpy(self.data_sdp_num[index]))

                sdp_no_trucation = sdp_num[i].item()
                if sdp_no_trucation > self.max_entity_num:
                    sdp_no_trucation = self.max_entity_num
                sdp_nums.append(sdp_no_trucation)

                L = len(ins["vertexSet"])
                titles.append(ins["title"])

                vertexSets.append(ins["vertexSet"])

                j = 0
                for h_idx in range(L):
                    for t_idx in range(L):
                        if h_idx != t_idx:
                            hlist = ins["vertexSet"][h_idx]
                            tlist = ins["vertexSet"][t_idx]

                            for h in hlist:
                                h_mapping[i, j, h["pos"][0] : h["pos"][1]] = (
                                    1.0 / len(hlist) / (h["pos"][1] - h["pos"][0])
                                )
                            for t in tlist:
                                t_mapping[i, j, t["pos"][0] : t["pos"][1]] = (
                                    1.0 / len(tlist) / (t["pos"][1] - t["pos"][0])
                                )

                            relation_mask[i, j] = 1

                            delta_dis = hlist[0]["pos"][0] - tlist[0]["pos"][0]

                            if delta_dis < 0:
                                ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                            else:
                                ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])
                            j += 1

                max_h_t_cnt = max(max_h_t_cnt, j)
                label_set = {}
                for label in ins["labels"]:
                    label_set[(label["h"], label["t"], label["r"])] = label["in" + TRAIN_PREFIX]

                labels.append(label_set)

                L_vertex.append(L)
                indexes.append(index)

            input_lengths = (context_idxs[:cur_bsz] > 0).long().sum(dim=1)
            max_c_len = int(input_lengths.max())

            entity_mention_num = list(map(add, entity_num, node_num[:cur_bsz].squeeze(1).tolist()))
            max_sdp_num = max(sdp_nums)
            all_node_num = list(map(add, sdp_nums, entity_mention_num))

            max_entity_num = max(entity_num)
            max_sentence_num = max(sentence_num)
            b_max_mention_num = int(
                node_num[:cur_bsz].max()
            )  # - max_entity_num - max_sentence_num
            all_node_num = torch.LongTensor(all_node_num)

            yield {
                "context_idxs": context_idxs[:cur_bsz, :max_c_len].contiguous(),
                "context_pos": context_pos[:cur_bsz, :max_c_len].contiguous(),
                "h_mapping": h_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                "t_mapping": t_mapping[:cur_bsz, :max_h_t_cnt, :max_c_len],
                "context_seg": context_seg[:cur_bsz, :max_c_len].contiguous(),
                "labels": labels,
                "L_vertex": L_vertex,
                "input_lengths": input_lengths,
                "context_ner": context_ner[:cur_bsz, :max_c_len].contiguous(),
                "context_char_idxs": context_char_idxs[:cur_bsz, :max_c_len].contiguous(),
                "relation_mask": relation_mask[:cur_bsz, :max_h_t_cnt],
                "titles": titles,
                "ht_pair_pos": ht_pair_pos[:cur_bsz, :max_h_t_cnt],
                "node_position": node_position[
                    :cur_bsz, :b_max_mention_num, :max_c_len
                ].contiguous(),
                "node_sent_num": node_sent_num[:cur_bsz, :max_sentence_num].contiguous(),
                "entity_position": entity_position[
                    :cur_bsz, :max_entity_num, :max_c_len
                ].contiguous(),
                "indexes": indexes,
                "all_node_num": all_node_num,
                "entity_num": entity_num,
                "sent_num": sentence_num,
                "sdp_position": sdp_position[:cur_bsz, :max_sdp_num, :max_c_len].contiguous(),
                "sdp_num": sdp_nums,
                "vertexsets": vertexSets,
            }
