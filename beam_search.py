import numpy as np
import torch
from nmt.utilities import pause


class Beam:
    def __init__(self, width, depth, vocabulary, device):
        self.decoder_forward = None
        self.width = width
        self.depth = depth
        self.vocabulary = vocabulary
        self.hypothesis = {}
        self.device = device

        self.best_sequence = None
        self.candidates_score = None
        self.prev_hidden_states = None
        # self.decoder = decoder
        # self.attention_tensor = None
        self.n_hypothesis = None
        self.minus_inf = torch.Tensor([float("-Inf")]).to(self.device)

    def decode(self, init_s, init_y, decoder_forward):
        """
                                ----------
                @params

                    init_s: tensor, with dimension (dir*layers, batch_size, n_features)
                    init_y: tensor, with dimension (batch_size)
                    attention_tensor: tensor, with dimension (batch_size, ?, n_features)
                @return
                 ----------
                """

        # self.attention_tensor = attention_tensor
        self.decoder_forward = decoder_forward
        self.hypothesis = {}
        decoder_output, decoder_hidden, _ = self.decoder_forward(init_y, init_s)
        decoder_output = decoder_output.squeeze(1)
        decoder_output = decoder_output.detach()
        decoder_hidden = decoder_hidden.detach()
        # print("Decoder output:", len(decoder_output.size()))

        # if len(decoder_output.size()) == 1:
        #     decoder_output = torch.unsqueeze(decoder_output, 1)

        self.prev_hidden_states = torch.zeros([self.width] + list(decoder_hidden.size()))
        self.prev_hidden_states = decoder_hidden.expand_as(self.prev_hidden_states).clone()
        # print("self.prev_hidden_states: ", self.prev_hidden_states.size())
        #  self.prev_hidden_states tensor with dim (n_candidates, dir*layers, batch_size, n_features)

        self.candidates_score, self.best_sequence = torch.sort(decoder_output, descending=True, dim=1)
        # print("Initial score size1: ", self.candidates_score.size())
        self.candidates_score = self.candidates_score[:, :self.width]
        # print("Initial score size2: ", self.candidates_score.size())
        # self.candidates_score = torch.log(self.candidates_score[:, :self.width])
        # self.candidates_score tensor with dim (batch_size, n_candidates)

        self.best_sequence = self.best_sequence[:, :self.width]

        # print("Initial score2: ", self.candidates_score)
        # print("Best seq", self.best_sequence[0])
        # print("Argmax: ", np.argmax(decoder_output.detach().cpu().numpy(), axis=1)[0])

        self.best_sequence = torch.unsqueeze(self.best_sequence, 2)
        # print(self.best_sequence.size())
        # self.best_sequence tensor with dim (batch_size, n_candidates, depth)

        # self.best_sequence = torch.cat([self.best_sequence, self.best_sequence], dim=2)

        # last_letters = self.best_sequence[:, :, -1]

        batch_size = self.best_sequence.size(0)
        self.n_hypothesis = batch_size * [0]

        for i in range(batch_size):
            self.hypothesis[i] = []

        for i, value in enumerate(self.n_hypothesis):
            if value > 0:
                self.candidates_score[i, self.width - value:] = self.minus_inf
        #
        for d in range(1, self.depth):
            self.generate_candidates()

            stop = True
            for i in range(batch_size):
                stop = stop and (self.n_hypothesis[i] == self.width)
                # print(f"Sentence {i}", len(self.hypothesis[i]),
                #       len(self.hypothesis[i]) == self.width, stop)

            if stop:
                # print("Break point is achieved")
                break

            if d == self.depth - 1:
                # print("Max lenght is achieved")
                for i in range(batch_size):
                    if self.n_hypothesis[i] == 0:
                        sequence = self.vocabulary.tensor_to_line(self.best_sequence[i, 0, :])
                        score = self.candidates_score[i, 0].detach().cpu().numpy()
                        score = np.asscalar(score)
                        self.hypothesis[i].append([sequence, score])
                        # print("Add last best hypothesis")

                # for i in range(batch_size):
                #     if len(self.hypothesis[i]) == 0:
                #         # sequence = self.vocabulary.tensor_to_line(self.best_sequence[index[0], index[1], :])
                #         # score = self.candidates_score[index[0], index[1]].detach().cpu().numpy()
                #         # score = np.asscalar(score)
                #
                #         sequence = self.vocabulary.tensor_to_line(self.best_sequence[i, 0, :])
                #         score = self.candidates_score[i, 0].detach().cpu().numpy()
                #         score = np.asscalar(score)
                #         self.hypothesis[i].append([sequence, score])
                #         print("End of the sentence is not found")

        # #print(self.best_sequence[0, 0, :])

        batch_size = self.best_sequence.size(0)

        # print(self.hypothesis)
        # print(self.best_sequence[0, 0, :])
        # print("___________________")
        # print(self.best_sequence.size())
        # print(self.vocabulary.tensor_to_line(self.best_sequence[0, 0, :]))
        # print(self.vocabulary.tensor_to_line(self.best_sequence[0, 1, :]))
        # print(self.vocabulary.tensor_to_line(self.best_sequence[0, 2, :]))
        # print(self.vocabulary.tensor_to_line(self.best_sequence[0, 3, :]))
        #
        # print(self.hypothesis[0])
        # print(self.hypothesis[1])
        #
        # print("___________________")

        output = []
        for i in range(batch_size):
            output.append(sorted(self.hypothesis[i], key=lambda x: x[1], reverse=True)[0][0])

        return output

    def generate_candidates(self):
        # self.best_sequence tensor with dim (batch_size, n_candidates, depth)
        # self.prev_hidden_states tensor with dim (n_candidates, dir*layers, batch_size, n_features)
        # self.candidates_score tensor with dim (batch_size, n_candidates)

        # current_hidden_states = torch.zeros(self.prev_hidden_states.size()).to(self.device)
        current_hidden_states = self.prev_hidden_states.clone()
        # print("Hidden size: ", current_hidden_states.size())

        current_score = None
        # print(current_hidden_states)

        candidates = None
        index_to_state = None

        seq_len = self.best_sequence.size(2)
        batch_size = self.best_sequence.size(0)

        for i in range(self.width):
            decoder_output, decoder_hidden, _ = \
                self.decoder_forward(self.best_sequence[:, i, -1].unsqueeze(0), self.prev_hidden_states[i])

            decoder_output = decoder_output.squeeze(1)

            decoder_output = decoder_output.detach()
            decoder_hidden = decoder_hidden.detach()

            # if len(decoder_output.size()) == 1:
            #     decoder_output = torch.unsqueeze(decoder_output, 1)

            current_hidden_states[i] = decoder_hidden

            # print("Equality: ", torch.all(torch.eq(current_hidden_states, self.prev_hidden_states)))

            score, generated_candidates = torch.sort(decoder_output, descending=True, dim=1)

            score = score[:, :self.width]
            generated_candidates = generated_candidates[:, :self.width]

            # prev_score = torch.unsqueeze(self.candidates_score[:, i], dim=1).expand_as(score)

            # print(self.candidates_score[:, [i]].size(), score.size())
            #
            # print("first sum ", seq_len * score + prev_score)
            # print("Second sum ", seq_len * score + self.candidates_score[:, [i]])
            # pause("SCORE")

            # add score back
            # score = (seq_len * score + prev_score) / (seq_len + 1)
            score = (score + seq_len*self.candidates_score[:, [i]]) / (seq_len + 1)
            # score = (seq_len * score + self.candidates_score[:, [i]]) / (seq_len + 1)
            # score = score + self.candidates_score[:, [i]]
            # # print("Initial score3: ", score)
            # print("Best seq3", generated_candidates[0])
            # print("Argmax3: ", np.argmax(decoder_output.detach().cpu().numpy(), axis=1)[0])

            current_score = torch.cat([current_score, score], dim=1) \
                if current_score is not None else score

            candidates = torch.cat([candidates, generated_candidates], dim=1) \
                if candidates is not None else generated_candidates

            index_to_state = torch.cat([index_to_state, i * torch.ones(score.size())], dim=1) \
                if index_to_state is not None else torch.zeros(score.size())
            #
            # print("Current candidates:", candidates[0])
            # print("current score:", current_score[0])
            # print("index_to_state:", index_to_state[0])
            #
            # pause("Pause")

        index_to_state = index_to_state.to(self.device)

        current_score, indexes = torch.sort(current_score, descending=True, dim=1)

        # print("One dim gather1:", torch.gather(candidates[0], 0, indexes[0]))
        # print("One dim gather2:", torch.gather(index_to_state[0], 0, indexes[0]))
        #
        # print("One dim scatter1:", candidates[0].scatter_(0, indexes[0], candidates[0]))
        # print("One dim scatter1:", candidates[0].scatter_(0, indexes[0], candidates[0]))
        # print("One dim scatter1:", candidates[0].scatter_(0, indexes[0], candidates[0]))
        # print("One dim scatter2:", index_to_state[0].scatter_(0, indexes[0], index_to_state[0]))

        # candidates = candidates.scatter_(1, indexes, candidates)
        # index_to_state = index_to_state.scatter_(1, indexes, index_to_state)

        # print(candidates)
        # print(index_to_state)
        # print(indexes)
        # pause("Pause")

        candidates = torch.gather(candidates, 1, indexes)
        index_to_state = torch.gather(index_to_state, 1, indexes).long()

        # print("Sorted Current candidates:", candidates[0])
        # print("Sorted current score:", current_score[0])
        # print("Sorted index_to_state:", index_to_state[0])
        # print("Sorted index:", indexes[0])
        #
        # pause("Pause")

        candidates = candidates[:, :self.width]


        index_to_state = index_to_state[:, :self.width]

        for i in range(batch_size):
            flatten_index = index_to_state[i]
            # print("flatten_index", flatten_index)
            # print("flatten_index", flatten_index.size())
            # pause("Pause")

            # current_hidden_states[:, :, i, :] = torch.index_select(current_hidden_states[:, :, i, :], 0, flatten_index)
            current_hidden_states[:, :, i, :] = current_hidden_states[flatten_index, :, i, :]

            # print(self.best_sequence.size(), current_hidden_states.size(), current_hidden_states[:, :, i, :].size(),
            #       flatten_index.size())
            # pause("Pause")
            # print("Best seq1", self.best_sequence[i])
            self.best_sequence[i] = self.best_sequence[i][flatten_index]
            # self.best_sequence[i] = torch.index_select(self.best_sequence[i], 0, flatten_index)
            # print("Best seq2", self.best_sequence[i])
        # print(candidates.size(), index_to_state.size(), current_hidden_states.size())
        self.prev_hidden_states = current_hidden_states.clone()
        self.candidates_score = current_score[:, :self.width]

        for i, value in enumerate(self.n_hypothesis):
            if value > 0:
                self.candidates_score[i, self.width - value:] = self.minus_inf
                # print(i, value)
        # print(self.candidates_score)
        # pause("")


        # print("Initial score2: ", self.candidates_score)
        candidates = torch.unsqueeze(candidates, 2)
        # end_of_sentence = (candidates == self.vocabulary.char_to_index[self.vocabulary.end_char]).nonzero()
        # print("end_of_sentence", end_of_sentence, end_of_sentence.size(), candidates.size(), candidates[end_of_sentence])
        self.best_sequence = torch.cat([self.best_sequence, candidates], dim=2)

        # print("OUTPUT TENSOR: ", self.best_sequence[0, 0])
        # pause("pause")

        self.add_hypothesis(candidates)

        # print("Candidates: ", candidates.size())
        # print("Best seq: ", self.best_sequence.size())

    def calculate_score(self):
        pass

    def add_hypothesis(self, candidates):
        # self.best_sequence tensor with dim (batch_size, n_candidates, depth)
        # print("End char: ", self.vocabulary.end_char)
        eos = (candidates == self.vocabulary.special_char_to_index[self.vocabulary.end_char]).nonzero()
        if eos.nelement() > 0:
            for i in range(eos.size(0)):
                index = list(eos[i])
                sentence_i = np.asscalar(index[0].detach().cpu().numpy())
                width_i = np.asscalar(index[1].detach().cpu().numpy())

                # print("hypothesis is valid", index)
                # print("end_of_sentence", eos, eos.size(), self.best_sequence.size())

                # print(candidates[list(eos[0])])
                sequence = self.vocabulary.tensor_to_line(self.best_sequence[sentence_i, width_i, :])
                score = self.candidates_score[sentence_i, width_i].detach().cpu().numpy()
                score = np.asscalar(score)
                if sequence.count(self.vocabulary.end_char) == 1 \
                        and self.n_hypothesis[sentence_i] < self.width \
                        and not torch.equal(
                        self.candidates_score[sentence_i, width_i],
                        self.minus_inf):

                        self.hypothesis[sentence_i].append([sequence, score])
                        self.n_hypothesis[sentence_i] += 1
                        # if sentence_index == 0:
                        #     print(f"hypothesis{sentence_index}", self.hypothesis[sentence_index][-1])
                    # print(sequence, score)
                self.candidates_score[sentence_i, width_i] = self.minus_inf
                # print("candidate score: ", self.candidates_score[sentence_i, width_i])
                # print(torch.equal(self.candidates_score[sentence_i, width_i],
                #                   self.minus_inf))
                # print("candidate score: ", self.candidates_score.size())
                # pause("")

                # print(sequence)
