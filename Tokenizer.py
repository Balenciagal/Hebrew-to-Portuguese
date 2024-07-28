import pickle
import tensorflow as tf
from transformers import BertTokenizer


PAD = 0


class Tokenizer:
    def __init__(self, data_to_fit, isLoad=False):
        """Initilize the tokenizer object

        Args:
            data_to_fit (_type_): the data to fit on the tokenizer / the file path to load if loading is choosen
            isLoad (bool, optional): If the data_to_fit is a file path to load. Defaults to False.
        """
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            oov_token="[?]",
        )
        if isLoad:  # if load option load the file instead
            self.load(data_to_fit)
            return
        self.tokenizer.fit_on_texts(
            [n.numpy().decode("utf-8") for n in list(data_to_fit.map(lambda x: x))]
        )
        self.tokenizer.index_word[PAD] = "[PAD]"
        self.tokenizer.word_index[self.tokenizer.index_word[PAD]] = PAD
        self.START = self.add_token("[START]")
        self.END = self.add_token("[END]")

    def add_token(self, string):
        index = max(self.tokenizer.index_word.keys()) + 1
        self.tokenizer.index_word[index] = string
        self.tokenizer.word_index[string] = index
        return index

    def tokenize(self, x):
        return tf.ragged.stack(
            list(
                map(
                    lambda x: tf.concat([[self.START], x, [self.END]], 0),
                    self.tokenizer.texts_to_sequences(
                        map(lambda x: x.decode("utf-8"), x.numpy())
                    ),
                )
            )
        )

    def detokenize(self, d):
        return tf.ragged.stack(
            self.tokenizer.sequences_to_texts(map(lambda x: x.numpy(), d))
        )

    def lookup_one(self, x):
        return tf.convert_to_tensor([self.tokenizer.index_word[i] for i in x.numpy()])

    def lookup(self, x):
        return tf.ragged.stack(list(map(lambda indexes: self.lookup_one(indexes), x)))

    def word_count(self):
        return len(self.tokenizer.word_index)

    def save(self, filename: str):
        with open(filename + ".pkl", "wb") as f:  # save the model
            pickle.dump(
                (
                    self.tokenizer.index_word,
                    self.tokenizer.word_index,
                    self.START,
                    self.END,
                ),
                f,
            )

    def load(self, file: str):
        with open(file, "rb") as f:  # load the model
            (
                self.tokenizer.index_word,
                self.tokenizer.word_index,
                self.START,
                self.END,
            ) = pickle.load(f)
