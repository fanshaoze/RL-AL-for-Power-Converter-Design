import torch
import torch.nn.functional as F
import math
from Models import nopeak_mask, create_masks

def init_vars(image0, image1, model, opt, vocab, image0_attribute, image1_attribute):
    
    init_tok = vocab.word2idx['<start>']
    # src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    image0 = model.cnn1(image0)

    image1 = model.cnn2(image1)

    if model.add_attribute:

        # image0_label = model.attribute_embedding1(image0_label)

        # image1_label = model.attribute_embedding2(image1_label)

        # joint_encoding = model.joint_encoding(torch.cat((image0, image0_label),1), \
        #                                     torch.cat((image1,image1_label),1))#independent cnn
        # if self.add_attribute:

        image0_attribute = model.attribute_embedding1(image0_attribute)

        image1_attribute = model.attribute_embedding2(image1_attribute)

        #joint_encoding = self.joint_encoding(torch.cat((image0, image0_attribute),1), torch.cat((image1,image1_attribute),1))
        joint_encoding = model.joint_encoding(image0, image1)

        joint_encoding = torch.cat((joint_encoding, image0_attribute), 1)

        joint_encoding = torch.cat((joint_encoding, image1_attribute), 1)

        # joint_encoding = model.bn(joint_encoding.transpose(1,2)).transpose(1,2)

    else:
        joint_encoding = model.joint_encoding(image0, image1)

    e_output = model.encoder(joint_encoding)
    
    outputs = torch.LongTensor([[init_tok]]).to(opt.device)
    
    trg_mask = nopeak_mask(1).to(opt.device)
    
    out = model.out(model.decoder(outputs, e_output, trg_mask))# (batch_size, seq_len, vocab_size)

    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.beam_size)

    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.beam_size, opt.max_seq_len).long().to(opt.device)

    outputs[:, 0] = init_tok

    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.beam_size, e_output.size(-2),e_output.size(-1)).to(opt.device)
    
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)

    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)

    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(image0, image1, model, opt, vocab, image0_label, image1_label):
    

    outputs, e_outputs, log_scores = init_vars(image0, image1, model, opt, vocab, image0_label, image1_label)
    eos_tok = vocab.word2idx['<end>']
    ind = None
    for i in range(2, opt.max_seq_len):
    
        trg_mask = nopeak_mask(i).to(opt.device)

        out = model.out(model.decoder(outputs[:,:i], e_outputs, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.beam_size)
        
        ones = (outputs==eos_tok).nonzero() # Occurrences of end symbols for all input sentences.

        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.beam_size:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        # length = (outputs[0]==eos_tok).nonzero()[0]
        # return ' '.join([vocab.idx2word[str(tok.item())] for tok in outputs[0][1:length]])
        return ' '.join([vocab.idx2word[str(tok.item())] for tok in outputs[0][1:]])
    
    else:
        length = (outputs[ind]==eos_tok).nonzero()[0]
        return ' '.join([vocab.idx2word[str(tok.item())] for tok in outputs[ind][1:length]])




# from operator import itemgetter

# import torch
# import torch.nn as nn

# import data_loader

# LENGTH_PENALTY = 1.2
# MIN_LENGTH = 5

# class SingleBeamSearchSpace():

#     def __init__(self, hidden, h_t_tilde = None, beam_size = 5, max_length = 255):
#         self.beam_size = beam_size
#         self.max_length = max_length

#         super(SingleBeamSearchSpace, self).__init__()

#         self.device = hidden[0].device
#         self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
#         self.prev_beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
#         self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
#         self.masks = [torch.ByteTensor(beam_size).zero_().to(self.device)] # 1 if it is done else 0

#         # |hidden[0]| = (n_layers, 1, hidden_size)
#         self.prev_hidden = torch.cat([hidden[0]] * beam_size, dim = 1)
#         self.prev_cell = torch.cat([hidden[1]] * beam_size, dim = 1)
#         # |prev_hidden| = (n_layers, beam_size, hidden_size)
#         # |prev_cell| = (n_layers, beam_size, hidden_size)

#         # |h_t_tilde| = (batch_size = 1, 1, hidden_size)
#         self.prev_h_t_tilde = torch.cat([h_t_tilde] * beam_size, dim = 0) if h_t_tilde is not None else None
#         # |prev_h_t_tilde| = (beam_size, 1, hidden_size)

#         self.current_time_step = 0
#         self.done_cnt = 0

#     def get_length_penalty(self, length, alpha = LENGTH_PENALTY, min_length = MIN_LENGTH):
#         p = (1 + length) ** alpha / (1 + min_length) ** alpha

#         return p

#     def is_done(self):
#         if self.done_cnt >= self.beam_size:
#             return 1
#         return 0

#     def get_batch(self):
#         y_hat = self.word_indice[-1].unsqueeze(-1)
#         hidden = (self.prev_hidden, self.prev_cell)
#         h_t_tilde = self.prev_h_t_tilde

#         # |y_hat| = (beam_size, 1)
#         # |hidden| = (n_layers, beam_size, hidden_size)
#         # |h_t_tilde| = (beam_size, 1, hidden_size) or None
#         return y_hat, hidden, h_t_tilde

#     def collect_result(self, y_hat, hidden, h_t_tilde):
#         # |y_hat| = (beam_size, 1, output_size)
#         # |hidden| = (n_layers, beam_size, hidden_size)
#         # |h_t_tilde| = (beam_size, 1, hidden_size)
#         output_size = y_hat.size(-1)

#         self.current_time_step += 1

#         cumulative_prob = y_hat + self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf')).view(-1, 1, 1).expand(self.beam_size, 1, output_size)
#         top_log_prob, top_indice = torch.topk(cumulative_prob.view(-1), self.beam_size, dim = -1)
#         # |top_log_prob| = (beam_size)
#         # |top_indice| = (beam_size)
#         self.word_indice += [top_indice.fmod(output_size)]
#         self.prev_beam_indice += [top_indice.div(output_size).long()]

#         self.cumulative_probs += [top_log_prob]
#         self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)]
#         self.done_cnt += self.masks[-1].float().sum()

#         self.prev_hidden = torch.index_select(hidden[0], dim = 1, index = self.prev_beam_indice[-1]).contiguous()
#         self.prev_cell = torch.index_select(hidden[1], dim = 1, index = self.prev_beam_indice[-1]).contiguous()
#         self.prev_h_t_tilde = torch.index_select(h_t_tilde, dim = 0, index = self.prev_beam_indice[-1]).contiguous()

#     def get_n_best(self, n = 1):
#         sentences = []
#         probs = []
#         founds = []

#         for t in range(len(self.word_indice)):
#             for b in range(self.beam_size):
#                 if self.masks[t][b] == 1:
#                     probs += [self.cumulative_probs[t][b] / self.get_length_penalty(t)]
#                     founds += [(t, b)]

#         for b in range(self.beam_size):
#             if self.cumulative_probs[-1][b] != -float('inf'):
#                 if not (len(self.cumulative_probs) - 1, b) in founds:
#                     probs += [self.cumulative_probs[-1][b]]
#                     founds += [(t, b)]

#         sorted_founds_with_probs = sorted(zip(founds, probs), 
#                                             key = itemgetter(1), 
#                                             reverse = True
#                                             )[:n]
#         probs = []

#         for (end_index, b), prob in sorted_founds_with_probs:
#             sentence = []

#             for t in range(end_index, 0, -1):
#                 sentence = [self.word_indice[t][b]] + sentence
#                 b = self.prev_beam_indice[t][b]

#             sentences += [sentence]
#             probs += [prob]

#         return sentences, probs

# class BeamSearchNode(object):
#     def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
#         '''
#         :param hiddenstate:
#         :param previousNode:
#         :param wordId:
#         :param logProb:
#         :param length:
#         '''
#         self.h = hiddenstate
#         self.prevNode = previousNode
#         self.wordid = wordId
#         self.logp = logProb
#         self.leng = length

#     def eval(self, alpha=1.0):
#         reward = 0
#         # Add here a function for shaping a reward

#         return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

# def batch_beam_search(image0, image1, model, opt, vocab, image0_label, image1_label):

#     decoded_batch = []
#     # decoding goes sentence by sentence
#     for idx in range(image0.size(0)):

#     # mask = None
#     # x_length = None
#     # if isinstance(src, tuple):
#     #     x, x_length = src
#     #     mask = self.generate_mask(x, x_length)
#     #     # |mask| = (batch_size, length)
#     # else:
#     #     x = src
#     # batch_size = x.size(0)

#     # emb_src = self.emb_src(x)
#     # h_src, h_0_tgt = self.encoder((emb_src, x_length))
#     # # |h_src| = (batch_size, length, hidden_size)
#     # h_0_tgt, c_0_tgt = h_0_tgt
#     # h_0_tgt = h_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
#     # c_0_tgt = c_0_tgt.transpose(0, 1).contiguous().view(batch_size, -1, self.hidden_size).transpose(0, 1).contiguous()
#     # # |h_0_tgt| = (n_layers, batch_size, hidden_size)
#     # h_0_tgt = (h_0_tgt, c_0_tgt)

#     # # initialize beam-search.
#     # spaces = [SingleBeamSearchSpace((h_0_tgt[0][:, i, :].unsqueeze(1), 
#     #                                     h_0_tgt[1][:, i, :].unsqueeze(1)), 
#     #                                     None, 
#     #                                     beam_size, 
#     #                                     max_length = max_length
#     #                                     ) for i in range(batch_size)]
#     # done_cnt = [space.is_done() for space in spaces]

#     # length = 0
#     # while sum(done_cnt) < batch_size and length <= max_length:
#     #     # current_batch_size = sum(done_cnt) * beam_size

#     #     # initialize fabricated variables.
#     #     fab_input, fab_hidden, fab_cell, fab_h_t_tilde = [], [], [], []
#     #     fab_h_src, fab_mask = [], []

#     #     # batchify.
#     #     for i, space in enumerate(spaces):
#     #         if space.is_done() == 0:
#     #             y_hat_, (hidden_, cell_), h_t_tilde_ = space.get_batch()

#     #             fab_input += [y_hat_]
#     #             fab_hidden += [hidden_]
#     #             fab_cell += [cell_]
#     #             if h_t_tilde_ is not None:
#     #                 fab_h_t_tilde += [h_t_tilde_]
#     #             else:
#     #                 fab_h_t_tilde = None

#     #             fab_h_src += [h_src[i, :, :]] * beam_size
#     #             fab_mask += [mask[i, :]] * beam_size

#     #     fab_input = torch.cat(fab_input, dim = 0)
#     #     fab_hidden = torch.cat(fab_hidden, dim = 1)
#     #     fab_cell = torch.cat(fab_cell, dim = 1)
#     #     if fab_h_t_tilde is not None:
#     #         fab_h_t_tilde = torch.cat(fab_h_t_tilde, dim = 0)
#     #     fab_h_src = torch.stack(fab_h_src)
#     #     fab_mask = torch.stack(fab_mask)
#     #     # |fab_input| = (current_batch_size, 1)
#     #     # |fab_hidden| = (n_layers, current_batch_size, hidden_size)
#     #     # |fab_cell| = (n_layers, current_batch_size, hidden_size)
#     #     # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
#     #     # |fab_h_src| = (current_batch_size, length, hidden_size)
#     #     # |fab_mask| = (current_batch_size, length)

#     #     emb_t = self.emb_dec(fab_input)
#     #     # |emb_t| = (current_batch_size, 1, word_vec_dim)

#     #     fab_decoder_output, (fab_hidden, fab_cell) = self.decoder(emb_t, fab_h_t_tilde, (fab_hidden, fab_cell))
#     #     # |fab_decoder_output| = (current_batch_size, 1, hidden_size)
#     #     context_vector = self.attn(fab_h_src, fab_decoder_output, fab_mask)
#     #     # |context_vector| = (current_batch_size, 1, hidden_size)
#     #     fab_h_t_tilde = self.tanh(self.concat(torch.cat([fab_decoder_output, context_vector], dim = -1)))
#     #     # |fab_h_t_tilde| = (current_batch_size, 1, hidden_size)
#     #     y_hat = self.generator(fab_h_t_tilde)
#     #     # |y_hat| = (current_batch_size, 1, output_size)

#     #     # separate the result for each sample.
#     #     cnt = 0
#     #     for space in spaces:
#     #         if space.is_done() == 0:
#     #             from_index = cnt * beam_size
#     #             to_index = (cnt + 1) * beam_size

#     #             # pick k-best results for each sample.
#     #             space.collect_result(y_hat[from_index:to_index],
#     #                                         (fab_hidden[:, from_index:to_index, :], 
#     #                                             fab_cell[:, from_index:to_index, :]),
#     #                                         fab_h_t_tilde[from_index:to_index]
#     #                                         )
#     #             cnt += 1

#     #     done_cnt = [space.is_done() for space in spaces]
#     #     length += 1

#     # # pick n-best hypothesis.
#     # batch_sentences = []
#     # batch_probs = []

#     # for i, space in enumerate(spaces):
#     #     sentences, probs = space.get_n_best(n_best)

#     #     batch_sentences += [sentences]
#     #     batch_probs += [probs]

#     return batch_sentences, batch_probs
