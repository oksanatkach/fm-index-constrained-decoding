from typing import List, Tuple, Optional
import torch
import requests


class IndexBasedLogitsProcessor:
    def __init__(
            self,
            num_beams: int,
            index_url: str = 'http://0.0.0.0:8000',
            pad_token_id: int = 0,
            eos_token_id: int = 2,
            force_decoding_from: Optional[List[int]] = None,
            stop_at_count: int = 0,
            always_allow_eos: bool = False,
            forced_bos_token_id: Optional[int] = None,
            end_marker: List[int] = None,
            length_reward_factor: float = 2.0,
            min_new_tokens: int = 5,
    ):
        super().__init__()
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
        self.BOOST = 0.0
        # self.BOOST = 20.0
        # self.BOOST = 10.0

        self.end_marker = end_marker
        self.index_url = index_url
        self.all_unigrams = torch.tensor(requests.get(f'{self.index_url}/occurring_distinct').json())
        self.length_reward_factor = length_reward_factor
        self.min_new_tokens = min_new_tokens

    def clone(self):
        return IndexBasedLogitsProcessor(
            num_beams=self._num_beams,
            index_url=self.index_url,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            force_decoding_from=self.force_decoding_from,
            stop_at_count=self.stop_at_count,
            always_allow_eos=self.always_allow_eos,
            forced_bos_token_id=self.forced_bos_token_id
        )

    def get_sub_sequence_count(self, sub_sequence: List[int]) -> int:
        response = requests.post(f'{self.index_url}/get_count', json={'sub_sequence': sub_sequence})
        return response.json()['count']

    def get_range(self, sequence: List[int]) -> int:
        response = requests.post(f'{self.index_url}/get_range', json={'sequence': sequence})
        return response.json()['range']

    def get_distinct_count_multi(self, lows, highs):
        response = requests.post(f'{self.index_url}/get_distinct_count_multi', json={'lows': lows, 'highs': highs})
        return response.json()['distinct_list']

    def get_trailing_corpus_ngram(self, sent: List[int]) -> List[int]:
        '''
        Find the longest ngram at the end of the generated sequence which matches the corpus
        :param sent:
        :return:
        '''
        # sent = self.remove_system_tokens(sent)
        # if sent == []:
        #     return []

        for ind in range(len(sent)-1, -1, -1):
            sub_sent = sent[ind:]
            if self.get_sub_sequence_count(sub_sent) == 0:
                return sent[ind+1:]
            return sent

    def __call__(self, input_ids: Tuple, scores: torch.FloatTensor) -> torch.Tensor:
        input_ids = torch.tensor([list(input_ids)])
        scores = scores.unsqueeze(0)

        mask = torch.full_like(scores, 0.0)

        if input_ids.size(1) == 0:
            mask[:, self.all_unigrams] = self.BOOST

        else:
            input_ids_list = input_ids.view(-1, self._num_beams, input_ids.shape[-1]).tolist()
            processed_input_ids_list = []
            for batch_id, beam_sent in enumerate(input_ids_list):
                batch_lst = []
                for beam_id, sent in enumerate(beam_sent):
                    # batch_lst.append(self.get_trailing_corpus_ngram(sent) if self.finished_thinking(sent) else None)
                    batch_lst.append(self.get_trailing_corpus_ngram(sent))
                processed_input_ids_list.append(batch_lst)

            lows = []
            highs = []
            fm_index_counts = []

            for batch_id, beam_sent in enumerate(processed_input_ids_list):
                for beam_id, sent in enumerate(beam_sent):
                    # check if finished thinking
                    if sent != None:
                        # check if not empty list
                        if sent:

                            if sent[-1] in (self.eos_token_id, self.pad_token_id):
                                low = 0
                                high = 0
                                count = 0

                            elif self.force_decoding_from is not None:
                                low, high = self.get_range(self.force_decoding_from + sent)
                                count = self.get_sub_sequence_count(self.force_decoding_from + sent)

                            else:
                                low, high = self.get_range(sent)

                                # how many continuations
                                count = self.get_sub_sequence_count(sent)

                            lows.append(low)
                            highs.append(high)
                            fm_index_counts.append(count)

            fm_index_result = self.get_distinct_count_multi(lows, highs)
            fm_index_result = fm_index_result[::-1]
            fm_index_counts = fm_index_counts[::-1]

            for batch_id, beam_sent in enumerate(processed_input_ids_list):
                for beam_id, sent in enumerate(beam_sent):
                    if sent != None:
                        if not sent:
                            # switching from free generation, start trying to copy a new ngram
                            distinct = torch.LongTensor(self.all_unigrams).to(scores.device)
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

                            # this needs to be optimized
                            # method 1
                            # additional_unigrams = [unigram for unigram in self.all_unigrams if unigram not in distinct]

                            # method 2
                            # distinct_set = set(distinct)
                            # additional_unigrams = [unigram for unigram in self.all_unigrams if unigram not in distinct_set]

                            distinct = torch.LongTensor(distinct).to(scores.device)

                            # method 3
                            mask_bool = ~torch.isin(self.all_unigrams, distinct)
                            additional_unigrams = self.all_unigrams[mask_bool]

                            # boost = self.BOOST + (len(sent) * self.length_reward_factor)
                            boost = self.BOOST * (self.length_reward_factor ** len(sent))
                            mask[batch_id * self._num_beams + beam_id, distinct] = boost

                            # allow new ngram generation
                            if distinct.shape[0] == 0:
                                boost = self.BOOST
                            else:
                                boost = self.BOOST / 2
                            mask[batch_id * self._num_beams + beam_id, additional_unigrams] = boost

                        # only boost the eos token if we have reached min_new_tokens
                        if self.always_allow_eos:
                            if input_ids.size(1) >= self.min_new_tokens:
                                # we need to boost <eos> proportionally to the sequence length to make it able to compete with the continuations above
                                # boost = self.BOOST + (input_ids.size(1) * self.length_reward_factor)
                                boost = self.BOOST * (self.length_reward_factor ** input_ids.size(1))
                                mask[batch_id * self._num_beams + beam_id, self.eos_token_id] = boost

        return scores + mask
