import numpy as np
from tqdm import tqdm
import json


def load(load_path):
    with open(load_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class HMM:
    def __init__(self):
        self.char2id = load('char2id.json')
        self.tag2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
        self.id2tag = {v: k for k, v in self.tag2id.items()}

        self.tag_size = len(self.tag2id)
        self.vocab_size = max(self.char2id.values()) + 1

        self.Pi = np.zeros(self.tag_size)
        self.A = np.zeros([self.tag_size, self.tag_size])
        self.B = np.zeros([self.tag_size, self.vocab_size])

        self.epsilon = 1e-8

    def get_Pi_A_B(self, data):
        for dic in tqdm(data):
            if dic['label'] == [] or dic['text'] == []:
                continue
            self.Pi[self.tag2id[dic['label'][0]]] += 1
            for index, tag in enumerate(dic['label'][1:]):
                this_tag = self.tag2id[tag]
                before_tag = self.tag2id[dic['label'][index]]
                self.A[before_tag][this_tag] += 1
        self.A[self.A == 0] = self.epsilon
        self.A /= np.sum(self.A, axis=0, keepdims=True)
        self.Pi[self.Pi == 0] = self.epsilon
        self.Pi /= np.sum(self.Pi)

        for dic in tqdm(data):
            for char, tag in zip(dic['text'], dic['label']):
                self.B[self.tag2id[tag]][self.char2id[char]] += 1
        self.B[self.B == 0] = self.epsilon
        self.B /= np.sum(self.B, axis=1, keepdims=True)

    def fit(self, data_path):
        train_data = load(data_path)
        self.get_Pi_A_B(train_data)

        self.Pi = np.log(self.Pi)
        self.A = np.log(self.A)
        self.B = np.log(self.B)
        print("DONE!")

    def viterbi_decode(self, text):
        seq_len = len(text)

        T1_table = np.zeros([self.tag_size, seq_len])
        T2_table = np.zeros([self.tag_size, seq_len])

        start_p_obs_state = self.get_p_obs(text[0])
        T1_table[:, 0] = self.Pi + start_p_obs_state
        T2_table[:, 0] = np.nan

        for i in range(1, seq_len):
            p_obs_state = self.get_p_obs(text[i])
            p_obs_state = np.expand_dims(p_obs_state, axis=0)
            prev_score = np.expand_dims(T1_table[:, i - 1], axis=-1)
            score = prev_score + self.A + p_obs_state

            T1_table[:, i] = np.max(score, axis=0)
            T2_table[:, i] = np.argmax(score, axis=0)

        best_tag_id = int(np.argmax(T1_table[:, -1]))
        best_tags = [best_tag_id, ]
        for i in range(seq_len-1, 0, -1):
            best_tag_id = int(T2_table[best_tag_id, i])
            best_tags.append(best_tag_id)
        return list(reversed(best_tags))

    def get_p_obs(self, char):
        char_token = self.char2id.get(char, 0)
        if char_token == 0:
            return np.log(np.ones(self.tag_size)/self.tag_size)
        return np.ravel(self.B[:, char_token])

    def predict(self, text):
        if len(text) == 0:
            raise NotImplementedError("输入文本为空!")
        best_tag_id = self.viterbi_decode(text)
        self.print_func(text, best_tag_id)

    def print_func(self, text, best_tags_id):
        for char, tag_id in zip(text, best_tags_id):
            print(char+"_"+self.id2tag[tag_id]+"|", end="")

    def load_weights(self, Pi_path, A_path, B_path):
        self.Pi = np.load(Pi_path)
        self.A = np.load(A_path)
        self.B = np.load(B_path)

    def save_weights(self, Pi_path, A_path, B_path):
        for data, path in zip([self.Pi, self.A, self.B], [Pi_path, A_path, B_path]):
            np.save(path, data)


if __name__ == '__main__':
    model = HMM()
    # model.fit('data.json')
    # model.save_weights('./Pi', './A', './B')
    model.load_weights('./Pi.npy', './A.npy', './B.npy')
    model.predict(
        '一九九八年,我在中国的河南吃美国的面包,北京在河南北面的山上,我在中国的河南吃美国的面包,真是举国欢庆、喜气洋洋、无法无天啊！')
