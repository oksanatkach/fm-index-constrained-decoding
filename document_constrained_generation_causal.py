import torch
from logits_processor_zoo.transformers.base import BaseLogitsProcessor
from typing import Optional, List
from index import FMIndex


class IndexBasedLogitsProcessor(BaseLogitsProcessor):
    def __init__(
            self,
            index: FMIndex,
            num_beams: int,
            model_name: str,
            pad_token_id: int = 0,
            eos_token_id: int = 2,
            force_decoding_from: Optional[List[int]] = None,
            stop_at_count: int = 0,
            always_allow_eos: bool = False,
            forced_bos_token_id: Optional[int] = None,
            length_reward_factor=2.0,
    ):
        super().__init__()
        self.index = index
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self._num_beams = num_beams
        self.log_odds_weight = 0.0
        self.force_decoding_from = force_decoding_from
        self.force_decoding_second_token = None
        self.block_initial_stopwords = False
        self.stop_at_count = stop_at_count
        self.always_allow_eos = always_allow_eos
        self.forced_bos_token_id = forced_bos_token_id

        self.length_reward_factor = length_reward_factor
        # self.BOOST = 0.0
        # self.BOOST = 20.0
        self.BOOST = 10.0

        self.model_name = model_name
        self.end_markers_map = {'qwen': [151645, 198, 151644,  77091,    198],
                            'llama': [78191, 128007, 271]}
        self.end_marker = self.end_markers_map['llama']
        if self.model_name.lower().startswith('qwen'):
            self.end_marker = self.end_markers_map['qwen']
        elif self.model_name.lower().startswith('llama'):
            self.end_marker = self.end_markers_map['llama']

    @staticmethod
    def remove_end_marker(input_ids, end_marker):
        end_marker_tensor = torch.tensor(end_marker, device=input_ids.device)
        marker_len = len(end_marker)
        result = []

        for row in input_ids:
            for i in range(len(row) - marker_len + 1):
                if torch.equal(row[i:i + marker_len], end_marker_tensor):
                    # Found the marker, return everything after it
                    result.append(row[i + marker_len:])

        return torch.stack(result)

    def get_trailing_corpus_ngram(self, sent: List[int]) -> List[int]:
        '''
        Find the longest ngram at the end of the generated sequence which matches the corpus
        :param sent:
        :return:
        '''
        for ind in range(len(sent)-1, -1, -1):
            sub_sent = sent[ind:]
            if self.index.get_count(sub_sent) == 0:
                return sent[ind+1:]
            return sent

    def _process(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        input_ids = self.remove_end_marker(input_ids, end_marker=self.end_marker)

        # mask = torch.full_like(scores, float('-inf'))
        mask = torch.full_like(scores, 0.0)

        if self.forced_bos_token_id is not None:
            if input_ids.size(1) == 0:
                mask[:, self.forced_bos_token_id] = self.BOOST
                return scores + mask
            else:
                input_ids = input_ids[:, 1:]

        if input_ids.size(1) == 0:

            distinct = self.index.occurring_distinct
            distinct = torch.LongTensor(distinct).to(scores.device)
            mask[:, distinct] = self.BOOST

        else:
            input_ids_list = input_ids.view(-1, self._num_beams, input_ids.shape[-1]).tolist()

            lows = []
            highs = []
            fm_index_counts = []

            input_ids_list = [[self.get_trailing_corpus_ngram(sent) for sent in beam_sent] for beam_sent in input_ids_list]

            for batch_id, beam_sent in enumerate(input_ids_list):
                for beam_id, sent in enumerate(beam_sent):
                    if sent:

                        if sent[-1] in (self.eos_token_id, self.pad_token_id):
                            low = 0
                            high = 0
                            count = 0

                        elif self.force_decoding_from is not None:
                            low, high = self.index.get_range(self.force_decoding_from + sent)
                            count = self.index.get_count(self.force_decoding_from + sent)

                        else:
                            low, high = self.index.get_range(sent)

                            # how many continuations
                            count = self.index.get_count(sent)

                        lows.append(low)
                        highs.append(high)
                        fm_index_counts.append(count)

            # get all possible unique tokens that can continue this ngram and their count
            # for instance, if the same token continues the ngram in 6 different spots in the corpus, the count is 6
            fm_index_result = self.index.get_distinct_count_multi(lows, highs)
            # reverse
            fm_index_result = fm_index_result[::-1]
            fm_index_counts = fm_index_counts[::-1]

            all_unigrams = self.index.occurring_distinct

            for batch_id, beam_sent in enumerate(input_ids_list):
                for beam_id, sent in enumerate(beam_sent):
                    if not sent:
                        # switching from free generation, start trying to copy a new ngram
                        distinct = torch.LongTensor(all_unigrams).to(scores.device)
                        mask[batch_id * self._num_beams + beam_id, distinct] = self.BOOST

                    else:

                        if self.stop_at_count > 0 and fm_index_counts[-1] <= self.stop_at_count:
                            fm_index_result.pop()
                            fm_index_counts.pop()
                            distinct = [self.eos_token_id]

                        elif sent[-1] == self.eos_token_id:
                            fm_index_result.pop()
                            fm_index_counts.pop()
                            distinct = [self.pad_token_id]

                        elif sent[-1] == self.pad_token_id:
                            fm_index_result.pop()
                            fm_index_counts.pop()
                            distinct = [self.pad_token_id]

                        else:
                            fm_index_counts.pop()
                            # all tokens that can continue this sequence
                            distinct, _ = fm_index_result.pop()

                        additional_unigrams = [unigram for unigram in all_unigrams if unigram not in distinct]
                        distinct = torch.LongTensor(distinct).to(scores.device)

                        # boost = self.BOOST + (len(sent) * self.length_reward_factor)
                        boost = self.BOOST * (self.length_reward_factor ** len(sent))
                        mask[batch_id * self._num_beams + beam_id, distinct] = boost

                        # allow new ngram generation
                        mask[batch_id * self._num_beams + beam_id, additional_unigrams] = self.BOOST / 2

        if self.always_allow_eos:
            # mask[:, self.eos_token_id] = self.BOOST
            # we need to boost <eos> proportionatly to the sequence length to make it able to compete with the continuations above
            # boost = self.BOOST + (input_ids.size(1) * self.length_reward_factor)
            boost = self.BOOST * (self.length_reward_factor ** input_ids.size(1))
            mask[:, self.eos_token_id] = boost

        return scores + mask
