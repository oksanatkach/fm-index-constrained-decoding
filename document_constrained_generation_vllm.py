from typing import List, Tuple, Optional
import torch
from index import FMIndex


class IndexBasedLogitsProcessor:
    def __init__(
            self,
            index: FMIndex,
            num_beams: int,
            pad_token_id: int = 0,
            eos_token_id: int = 2,
            force_decoding_from: Optional[List[int]] = None,
            stop_at_count: int = 0,
            always_allow_eos: bool = False,
            forced_bos_token_id: Optional[int] = None,
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
        self.BOOST = 0.0
        # self.BOOST = 20.0
        # self.BOOST = 10.0

    def clone(self):
        return IndexBasedLogitsProcessor(
            num_beams=self._num_beams,
            index=self.index,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            force_decoding_from=self.force_decoding_from,
            stop_at_count=self.stop_at_count,
            always_allow_eos=self.always_allow_eos,
            forced_bos_token_id=self.forced_bos_token_id
        )

    def __call__(self, input_ids: Tuple, scores: torch.FloatTensor) -> torch.Tensor:
        # reward the model if the ngram is longer

        input_ids = torch.tensor([list(input_ids)])
        scores = scores.unsqueeze(0)

        mask = torch.full_like(scores, float('-inf'))
        # mask = torch.full_like(scores, 0.0)

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
            # stage 2: glue multiple ngrams together
            # distinct_1_gram = self.index.occurring_distinct

            input_ids_list = input_ids.view(-1, self._num_beams, input_ids.shape[-1]).tolist()

            lows = []
            highs = []
            fm_index_counts = []

            for batch_id, beam_sent in enumerate(input_ids_list):
                for beam_id, sent in enumerate(beam_sent):

                    if sent[-1] in (self.eos_token_id, self.pad_token_id):
                        low = 0
                        high = 0
                        count = 0

                    elif self.force_decoding_from is not None:
                        low, high = self.index.get_range(self.force_decoding_from + sent)
                        count = self.index.get_count(self.force_decoding_from + sent[:-1])

                    else:
                        low, high = self.index.get_range(sent)
                        count = self.index.get_count(sent[:-1])

                    lows.append(low)
                    highs.append(high)
                    fm_index_counts.append(count)

            fm_index_result = self.index.get_distinct_count_multi(lows, highs)
            fm_index_result = fm_index_result[::-1]
            fm_index_counts = fm_index_counts[::-1]

            for batch_id, beam_sent in enumerate(input_ids_list):
                for beam_id, sent in enumerate(beam_sent):

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
                        distinct, _ = fm_index_result.pop()

                    # if distinct in distinct_1_gram:
                    #     distinct_1_gram.remove(distinct)
                    distinct = torch.LongTensor(distinct).to(scores.device)

                    mask[batch_id * self._num_beams + beam_id, distinct] = self.BOOST

            # distinct_1_gram = torch.LongTensor(distinct_1_gram).to(scores.device)
            # mask[:, distinct_1_gram] = 5.0
            # mask[:, distinct_1_gram] = 0.0

        if self.always_allow_eos:
            mask[:, self.eos_token_id] = self.BOOST

        return scores + mask
