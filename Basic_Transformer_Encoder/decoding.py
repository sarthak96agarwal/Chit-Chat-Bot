import math
from operator import attrgetter
import torch
num_gpus = torch.cuda.device_count()
if num_gpus > 0:
    current_device = 'cuda'
else:
    current_device = 'cpu'
    
def nucleus_sampling(model, batch, batch_size, threshold=0.9, use_packed=True):
    model.eval()
        
    text_vecs = batch['text_vecs'].to(current_device)
    if use_packed:
        encoded = model.encoder(text_vecs, batch['text_lens'], use_packed=batch['use_packed'])
    else:
        encoded = model.encoder(text_vecs)
    encoder_output, encoder_hidden, attention_mask = encoded
        
    # 1 is __start__
    starts = torch.Tensor([1]).long().to(model.decoder.embedding.weight.device).expand(batch_size, 1).long()  # expand to batch size
    decoder_hidden = encoder_hidden

    # greedy decoding here        
    preds = [starts]
    scores = []

    # track if each sample in the mini batch is finished
    # if all finished, stop predicting
    
    finish_mask = torch.Tensor([0]*batch_size).byte().to(model.decoder.embedding.weight.device)
    xs = starts
    _attn_w_log = []

    for ts in range(100):
        decoder_output, decoder_hidden, attn_w_log = model.decoder(xs, decoder_hidden, encoded)  # decoder_output: [batch, time, vocab]
        _probs = torch.softmax(decoder_output, dim=-1)[0][0]
        _sorted_probs,_sorted_indices = torch.sort(_probs, descending=True)
        cumulative_probs = torch.cumsum(_sorted_probs, dim=-1)
        selected_probs = cumulative_probs<threshold
        selected_probs[1:]=selected_probs[:-1].clone()
        selected_probs[0]=True
        _sorted_probs[~selected_probs] = 0
        P = _sorted_probs.sum()
        _sorted_probs /= P
        chosen_index = torch.multinomial(_sorted_probs, 1)
        _preds = _sorted_indices[chosen_index]
        _scores = torch.log(_probs[_preds])
        _preds = _preds.unsqueeze(0)

        preds.append(_preds)
        _attn_w_log.append(attn_w_log)
        scores.append(_scores.view(-1)*(finish_mask == 0).float())

        finish_mask += (_preds == 2).byte().view(-1)
        
        if not (torch.any(~finish_mask.bool())):
            break
        
        xs = _preds
    
    preds = torch.cat(preds, dim=-1)
        
    return preds,torch.sum(torch.Tensor(scores))


class _HypothesisTail(object):
    """Hold some bookkeeping about a hypothesis."""

    # use slots because we don't want dynamic attributes here
    __slots__ = ['timestep', 'hypid', 'score', 'tokenid']

    def __init__(self, timestep, hypid, score, tokenid):
        self.timestep = timestep
        self.hypid = hypid
        self.score = score
        self.tokenid = tokenid

class Beam(object):
    """
    This class serves to keep info about partial hypothesis and perform the beam step
    """
    
    def __init__(
        self,
        beam_size,
        padding_token=0,
        bos_token=1,
        eos_token=2,
        min_length=3,
        min_n_best=3,
        device='cpu',
        # for n-gram blocking below
        n=1,
        verbose=False
    ):
        
        self.beam_size = beam_size
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.device = device
        # recent score for each hypo in the beam
        self.scores = None
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)
        ]
        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.min_n_best = min_n_best
        self.partial_hyps = [[self.bos] for i in range(beam_size)]
    
        # n-gram blocking related below
        self.n = n
        self.banned_tokens = set()
        self.verbose = verbose
    def get_output_from_current_step(self):
        """Get the output at the current step."""
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """Get the backtrack at the current step."""
        return self.bookkeep[-1]
    
    ##################### N-GRAM BLOCKING PART START #####################
    
    def n_gram_block(self, active_hyp, n=1):

        banned_tokens = []
        
        if n!=1:
            l = active_hyp

            history = tuple(l[-(n-1):])

            for ngram in zip(*[l[i:] for i in range(n)]):
                if ngram[:-1] == history:
                    banned_tokens.append(ngram[-1])
        else:
            history = ''
            for token in active_hyp:
                if token not in banned_tokens:
                    banned_tokens.append(token)            
            
        if self.verbose:
            print(f'N: {n}')
            print(f'Active hyp: {active_hyp}')
            print(f'history: {history}')
            print(f'banned: {banned_tokens}')
        
        return list(set(banned_tokens))
    
    ##################### N-GRAM BLOCKING PART END ########################
    
    def select_paths(self, logprobs, prior_scores):
        """Select the next vocabulary item in these beams."""
        # beam search actually looks over all hypotheses together so we flatten
        beam_scores = logprobs + prior_scores.unsqueeze(1).expand_as(logprobs)
        
        # n-gramm\ blocking part
        current_length = len(self.all_scores)
        if current_length > 0:
            for hyp_id in range(beam_scores.size(0)):
                active_hyp = tuple(self.partial_hyps[hyp_id])
                banned_tokens = self.n_gram_block(active_hyp, n=self.n)
                if len(banned_tokens) > 0:
                    beam_scores[:, banned_tokens] = -10e5
            
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(flat_beam_scores, self.beam_size, dim=-1)
        voc_size = logprobs.size(-1)

        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs / voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size
        
        return (hyp_ids, tok_ids, best_scores)
    
    def advance(self, logprobs):
        """Advance the beam one step."""
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(logprobs.size(0)):
                logprobs[hyp_id][self.eos] = -10e5

        if self.scores is None:
            logprobs = logprobs[0:1]  # we use only the first hyp now, since they are all same
            self.scores = torch.zeros(1).type_as(logprobs).to(logprobs.device)
            
        hyp_ids, tok_ids, self.scores = self.select_paths(logprobs, self.scores)
        
        # clone scores here to avoid referencing penalized EOS in the future!
        self.all_scores.append(self.scores.clone())

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [
            self.partial_hyps[hyp_ids[i]] + [tok_ids[i].item()]
            for i in range(self.beam_size)
        ]

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                self.scores[hypid] = -10e5
                #  this is finished hypo, adding to finished
                eostail = _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.all_scores[-1][hypid],
                    tokenid=self.eos,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1
    
    def is_done(self):
        """Return whether beam search is complete."""
        return self.eos_top and self.n_best_counter >= self.min_n_best

    def get_top_hyp(self):
        """
        Get single best hypothesis.
        :return: hypothesis sequence and the final score
        """
        return self._get_rescored_finished(n_best=1)[0]

    def _get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.
        :param timestep:
            timestep with range up to len(self.outputs) - 1
        :param hyp_id:
            id with range up to beam_size - 1
        :return:
            hypothesis sequence
        """
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(
                _HypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback],
                    tokenid=self.outputs[i][endback],
                )
            )
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    def _get_pretty_hypothesis(self, list_of_hypotails):
        """Return hypothesis as a tensor of token ids."""
        return torch.stack([ht.tokenid for ht in reversed(list_of_hypotails)])

    def _get_rescored_finished(self, n_best=None, add_length_penalty=False):
        """
        Return finished hypotheses according to adjusted scores.
        Score adjustment is done according to the Google NMT paper, which
        penalizes long utterances.
        :param n_best:
            number of finalized hypotheses to return
        :return:
            list of (tokens, score) pairs, in sorted order, where:
              - tokens is a tensor of token ids
              - score is the adjusted log probability of the entire utterance
        """
        # if we never actually finished, force one
        if not self.finished:
            self.finished.append(
                _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=0,
                    score=self.all_scores[-1][0],
                    tokenid=self.eos,
                )
            )

        rescored_finished = []
        for finished_item in self.finished:
            if add_length_penalty:
                current_length = finished_item.timestep + 1
                # these weights are from Google NMT paper
                length_penalty = math.pow((1 + current_length) / 6, 0.65)
            else:
                length_penalty = 1
            rescored_finished.append(
                _HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                )
            )

        # Note: beam size is almost always pretty small, so sorting is cheap enough
        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        return [
            (self._get_pretty_hypothesis(self._get_hyp_from_finished(hyp)), hyp.score)
            for hyp in srted
        ]

def reorder_encoder_states(encoder_states, indices):
        """Reorder encoder states according to a new set of indices."""
        enc_out, hidden, attention_mask = encoder_states

        # LSTM or GRU/RNN hidden state?
        if isinstance(hidden, torch.Tensor):
            hid, cell = hidden, None
        else:
            hid, cell = hidden

        if not torch.is_tensor(indices):
            # cast indices to a tensor if needed
            indices = torch.LongTensor(indices).to(hid.device)

        hid = hid.index_select(1, indices)
        if cell is None:
            hidden = hid
        else:
            cell = cell.index_select(1, indices)
            hidden = (hid, cell)

        enc_out = enc_out.index_select(0, indices)
        attention_mask = attention_mask.index_select(0, indices)

        return enc_out, hidden, attention_mask
    
    
def reorder_decoder_incremental_state(incremental_state, inds):
    if torch.is_tensor(incremental_state):
        # gru or lstm
        return torch.index_select(incremental_state, 1, inds).contiguous()
    elif isinstance(incremental_state, tuple):
        return tuple(
            self.reorder_decoder_incremental_state(x, inds)
            for x in incremental_state)

def get_nbest_list_from_beam(beam, dictionary, n_best=None, add_length_penalty=False):
    if n_best is None:
        n_best = beam.min_n_best
    nbest_list = beam._get_rescored_finished(n_best=n_best, add_length_penalty=add_length_penalty)
    
    nbest_list_text = [(dictionary.v2t(i[0].cpu().tolist()), i[1].item()) for i in nbest_list]
    
    return nbest_list_text


def generate_with_beam(beam_size, min_n_best, model, batch, batch_size, n=1, verbose=False, use_packed=True):
    """
    This function takes a model, batch, beam settings and performs decoding with a beam
    """
    beams = [Beam(beam_size, min_n_best=min_n_best, eos_token=chat_dict.word2ind['__end__'], padding_token=chat_dict.word2ind['__null__'], bos_token=chat_dict.word2ind['__start__'], device=current_device, n=n, verbose=verbose) for _ in range(batch_size)]
    repeated_inds = torch.arange(batch_size).to(current_device).unsqueeze(1).repeat(1, beam_size).view(-1)
    
    text_vecs = batch['text_vecs'].to(current_device)
    if use_packed:
        encoder_states = model.encoder(text_vecs, batch['text_lens'], use_packed=batch['use_packed'])
    else:
        encoder_states = model.encoder(text_vecs)
    
    model.eval()
    
    encoder_states = reorder_encoder_states(encoder_states, repeated_inds)  # no actual reordering here, but repeating beam size times each sample in the minibatch
    encoder_output, encoder_hidden, attention_mask = encoder_states
    
    incr_state = encoder_hidden  # we init decoder hidden with last encoder_hidden
    
    # 1 is a start token id
    starts = torch.Tensor([1]).long().to(model.decoder.embedding.weight.device).expand(batch_size*beam_size, 1).long()  # expand to batch_size * beam_size
    decoder_input = starts
    
    with torch.no_grad():
        for ts in range(100):
            if all((b.is_done() for b in beams)):
                break
            score, incr_state, attn_w_log = model.decoder(decoder_input, incr_state, encoder_states)
            score = score[:, -1:, :]  # take last time step and eliminate the dimension
            score = score.view(batch_size, beam_size, -1)
            score = torch.log_softmax(score, dim=-1)
         
            for i, b in enumerate(beams):
                if not b.is_done():

                    b.advance(score[i])

            incr_state_inds = torch.cat([beam_size * i + b.get_backtrack_from_current_step() for i, b in enumerate(beams)])
            incr_state = reorder_decoder_incremental_state(incr_state, incr_state_inds)
            selection = torch.cat([b.get_output_from_current_step() for b in beams]).unsqueeze(-1)
            decoder_input = selection

    beam_preds_scores = [list(b.get_top_hyp()) for b in beams]

    if verbose:
        for bi in range(batch_size):
            print(f'batch {bi}')
            for i in get_nbest_list_from_beam(beams[bi], chat_dict, n_best=min_n_best):
                print(i)
    
    return beam_preds_scores, beams

def ngram_beam(beam_size, n_best_beam, model, batch, batch_size=1, n=1, verbose=True, verbose2=True, use_packed=True):
       
    outputs = []
    if use_packed:    
        beam_preds_scores, beams = generate_with_beam(beam_size, n_best_beam, model, batch, batch_size=batch_size, n=n, verbose=verbose, use_packed=use_packed)
    else:
        beam_preds_scores, beams = generate_with_beam(beam_size, n_best_beam, model, batch, batch_size=batch_size, n=n, verbose=verbose, use_packed=use_packed)
    
    outputs = (beam_preds_scores, beams)
        
    if verbose2:
        for bi in range(batch_size):            
            for j in get_nbest_list_from_beam(outputs[1][bi], chat_dict, n_best_beam):
                print(j)

    return outputs