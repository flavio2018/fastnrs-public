import re
import string
from collections import OrderedDict
import torchtext

torchtext.disable_torchtext_deprecation_warning()
from torchtext.vocab import vocab
from torchtext.transforms import ToTensor, VocabTransform

EOS = "."
SOS = "?"
PAD = "#"
SEP = "/"
HAL = "$"


class Vocabulary:

    def __init__(
        self,
        x_vocab_chars,
        y_vocab_chars,
        device,
        sos,
        eos,
        specials_in_y=True,
        specials_in_x=False,
        tokenizer="char",
    ):
        self.specials = {
            "sos": SOS,
            "sep": SEP,
            "eos": EOS,
            "pad": PAD,
            "hal": HAL,
        }
        if specials_in_y:
            specials_y = [SOS, EOS, PAD, HAL]
        else:
            specials_y = [PAD]
        if specials_in_x:
            specials_x = [SOS, EOS, PAD, HAL]
        else:
            specials_x = [PAD]
        self.x_vocab = vocab(
            OrderedDict([(c, 1) for c in x_vocab_chars]),
            specials=specials_x,
            special_first=False,
        )
        self.y_vocab = vocab(
            OrderedDict([(c, 1) for c in y_vocab_chars]),
            specials=specials_y,
            special_first=False,
        )
        self.x_vocab.set_default_index(self.x_vocab[PAD])
        self.y_vocab.set_default_index(self.y_vocab[PAD])
        self.x_vocab_trans = VocabTransform(self.x_vocab)
        self.y_vocab_trans = VocabTransform(self.y_vocab)
        self.x_to_tensor_trans = ToTensor(padding_value=self.x_vocab[PAD])
        self.y_to_tensor_trans = ToTensor(padding_value=self.y_vocab[PAD])
        self.device = device
        self.sos = sos
        self.eos = eos
        self.tokenizer = tokenizer
        self.specials_in_x = specials_in_x
        self.specials_in_y = specials_in_y

    def get_special_idx(self, special, x=True):
        assert special in self.specials, f"Special character not recognized: {special}."
        if x:
            assert (
                self.specials[special] in self.x_vocab
            ), f"Special character {self.specials[special]} is not in x_vocab."
            return self.x_vocab[self.specials[special]]
        else:
            assert (
                self.specials[special] in self.y_vocab
            ), f"Special character {self.specials[special]} is not in y_vocab."
            return self.y_vocab[self.specials[special]]

    def tokenize_sample(self, sample: str):
        if self.tokenizer == "char":
            return self._tokenize_char(sample)
        elif self.tokenizer == "listops":
            return self._tokenize_listops(sample)
        elif self.tokenizer == "arithmetic":
            return self._tokenize_arithmetic(sample)
        elif self.tokenizer == "arithmetic_textseg":
            return self._tokenize_arithmetic_textseg(sample)
        elif self.tokenizer == "algebra":
            return self._tokenize_algebra(sample)
        elif self.tokenizer == "algebra_textseg":
            return self._tokenize_algebra_textseg(sample)
        elif self.tokenizer == "logic":
            return self._tokenize_char(sample)
        elif self.tokenizer == "alltask":
            return self._tokenize_alltask(sample)
        elif self.tokenizer == "alltask_textseg":
            return self._tokenize_alltask_textseg(sample)

    @staticmethod
    def _tokenize_char(sample: str) -> list:
        return [c for c in sample]

    @staticmethod
    def _tokenize_listops(sample: str) -> list:
        listops_re = re.compile(r"(\d)|(SM|MIN|MAX)|([\[\]])|([?.#$/])")
        matches = listops_re.findall(sample)
        return [[submatch for submatch in match if submatch][0] for match in matches]

    @staticmethod
    def _tokenize_arithmetic(sample: str) -> list:
        arithmetic_re = re.compile(r"([+\-*])|([0-9]+)|([()])|([?.#$])")
        matches = arithmetic_re.findall(sample)
        return [[submatch for submatch in match if submatch][0] for match in matches]

    @staticmethod
    def _tokenize_arithmetic_textseg(sample: str) -> list:
        if set(list(sample)) <= {"0", "1"}:
            return [c for c in sample]
        arithmetic_re = re.compile(r"([+\-*])|([0-9]+)|([()])|([?.#$])")
        matches = arithmetic_re.findall(sample)
        return [[submatch for submatch in match if submatch][0] for match in matches]

    @staticmethod
    def _tokenize_algebra(sample: str) -> list:
        algebra_re = re.compile(
            r"([+\-*])|([0-9]+\*[abxy*]+)|([0-9]+)|([abxy*]+)|([()])|([?.#$])"
        )
        matches = algebra_re.findall(sample)
        return [[submatch for submatch in match if submatch][0] for match in matches]

    @staticmethod
    def _tokenize_algebra_textseg(sample: str) -> list:
        if set(list(sample)) <= {"0", "1"}:
            return [c for c in sample]
        algebra_re = re.compile(
            r"([+\-*])|([0-9]+\*[abxy*]+)|([0-9]+)|([abxy*]+)|([()])|([?.#$])"
        )
        matches = algebra_re.findall(sample)
        return [[submatch for submatch in match if submatch][0] for match in matches]

    @staticmethod
    def _tokenize_alltask(sample: str) -> list:
        listops_re = re.compile(r"(\d)|(SM|MIN|MAX)|([\[\]])|([?.#$/])")
        arithmetic_re = re.compile(r"([+\-*])|([0-9]+)|([()])|([?.#$])")
        algebra_re = re.compile(
            r"([+\-*])|([0-9]+\*[abxy*]+)|([0-9]+)|([abxy*]+)|([()])|([?.#$])"
        )

        task_name = Vocabulary.task_name_from_sample(
            sample.replace("?", "").replace(".", "")
        )
        if task_name == "listops":
            task_re = listops_re
        elif task_name == "logic":
            return [c for c in sample]
        elif task_name == "algebra":
            task_re = algebra_re
        elif task_name == "arithmetic":
            task_re = arithmetic_re
        else:
            assert False, "Unknown task {}".format(task_name)
        matches = task_re.findall(sample)
        return [[submatch for submatch in match if submatch][0] for match in matches]

    @staticmethod
    def _tokenize_alltask_textseg(sample: str) -> list:
        if set(list(sample)) <= {"0", "1"}:
            return [c for c in sample]
        listops_re = re.compile(r"(\d)|(SM|MIN|MAX)|([\[\]])|([?.#$/])")
        arithmetic_re = re.compile(r"([+\-*])|([0-9]+)|([()])|([?.#$])")
        algebra_re = re.compile(
            r"([+\-*])|([0-9]+\*[abxy*]+)|([0-9]+)|([abxy*]+)|([()])|([?.#$])"
        )

        task_name = Vocabulary.task_name_from_sample(
            sample.replace("?", "").replace(".", "")
        )
        if task_name == "listops":
            task_re = listops_re
        elif task_name == "logic":
            return [c for c in sample]
        elif task_name == "algebra":
            task_re = algebra_re
        elif task_name == "arithmetic":
            task_re = arithmetic_re
        else:
            assert False, "Unknown task {}".format(task_name)
        matches = task_re.findall(sample)
        return [[submatch for submatch in match if submatch][0] for match in matches]

    def str_to_batch(self, str_samples, x=True):
        if not x:
            if self.sos:
                str_samples = [f"{SOS}{sample}" for sample in str_samples]
            if self.eos:
                str_samples = [f"{sample}{EOS}" for sample in str_samples]

        string_tokenized_samples = [
            self.tokenize_sample(sample) for sample in str_samples
        ]

        if x:
            idx_tokenized_samples = self.x_vocab_trans(string_tokenized_samples)
            idx_padded_samples = self.x_to_tensor_trans(idx_tokenized_samples).to(
                self.device
            )
            return idx_padded_samples
        else:
            idx_tokenized_targets = self.y_vocab_trans(string_tokenized_samples)
            idx_padded_targets = self.y_to_tensor_trans(idx_tokenized_targets).to(
                self.device
            )
            return idx_padded_targets

    def batch_to_str(self, batch, x=True, replace_pad=True):
        vocab = self.x_vocab if x else self.y_vocab
        if not replace_pad:
            return ["".join(vocab.lookup_tokens(tokens)) for tokens in batch.tolist()]
        return [
            "".join(vocab.lookup_tokens(tokens)).replace(PAD, "")
            for tokens in batch.tolist()
        ]

    def cut_at_first_eos(self, output_str):
        if output_str.count(self.specials["eos"]) >= 1:
            position_first_eos = output_str.find(self.specials["eos"])
            return output_str[: position_first_eos + 1]
        else:
            return output_str

    @staticmethod
    def task_name_from_sample(sample):
        if (
            "MIN" in sample
            or "MAX" in sample
            or "SM" in sample
            or "[" in sample
            or "]" in sample
        ):
            return "listops"
        elif (
            "&" in sample
            or "|" in sample
            or "!" in sample
            or sample in string.ascii_lowercase + "TF"
        ):
            return "logic"
        elif "a" in sample or "b" in sample or "x" in sample or "y" in sample:
            return "algebra"
        else:
            return "arithmetic"

    @property
    def dataset_name(self):
        if self.tokenizer != "char" and self.tokenizer in [
            "listops",
            "arithmetic",
            "algebra",
            "logic",
            "alltask",
        ]:
            return self.tokenizer
        elif self.tokenizer == "arithmetic_textseg":
            return "arithmetic"
        elif self.tokenizer == "algebra_textseg":
            return "algebra"
        elif self.tokenizer == "alltask_textseg":
            return "alltask"
        elif self.tokenizer == "char":
            vocab_elems = self.x_vocab.get_itos()
            if (
                ("X" in vocab_elems or "S" in vocab_elems or "N" in vocab_elems)
                and ("&" in vocab_elems or "|" in vocab_elems or "!" in vocab_elems)
                and (
                    "a" in vocab_elems
                    or "b" in vocab_elems
                    or "x" in vocab_elems
                    or "y" in vocab_elems
                )
            ):
                return "alltask"
            elif "MIN" in vocab_elems or "MAX" in vocab_elems or "SM" in vocab_elems:
                return "listops"
            elif "&" in vocab_elems or "|" in vocab_elems or "!" in vocab_elems:
                return "logic"
            elif (
                "a" in vocab_elems
                or "b" in vocab_elems
                or "x" in vocab_elems
                or "y" in vocab_elems
            ):
                return "algebra"
            else:
                return "arithmetic"
        else:
            assert False, "Cannot recognize dataset from tokenizer {}".format(
                self.tokenizer
            )
