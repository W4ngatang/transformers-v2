# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" SuperGLUE processors and helpers """


import logging
import os

from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures, SpanClassificationExample, SpanClassificationFeatures


if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def tokenize_tracking_span(tokenizer, text, spans):
    """
    Tokenize while tracking what tokens spans (char idxs) get mapped to
    Strategy: split input around span, tokenize left of span, the span,
        and then recursively apply to remaning text + spans
    We assume spans are
        - inclusive on start and end
        - non-overlapping (TODO)
        - sorted (TODO)

    Args:

    Returns:

    """
    toks = tokenizer.encode_plus(text)
    full_toks = toks["input_ids"]
    prefix_len = len(tokenizer.decode(full_toks[:1])) + 1 # add a space
    len_covers = []
    for i in range(2, len(full_toks)):
        # iterate over the tokens and decode the length of the sequence
        # we start at 2 b/c 0 is empty; 1 is CLS/SOS
        partial_txt_len = len(tokenizer.decode(full_toks[:i], clean_up_tokenization_spaces=False))
        len_covers.append(partial_txt_len - prefix_len)

    span_locs = []
    for start, end in spans:
        start_tok, end_tok = None, None
        for tok_n, len_cover in enumerate(len_covers):
            if len_cover >= start and start_tok is None:
                start_tok = tok_n + 1 # account for [CLS] tok
            if len_cover >= end:
                assert start_tok is not None
                end_tok = tok_n + 1
                break
        assert start_tok is not None, "start_tok is None!"
        assert end_tok is not None, "end_tok is None!"
        span_locs.append((start_tok, end_tok))

    return toks, span_locs


def superglue_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: SuperGLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = superglue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = superglue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        if isinstance(example, SpanClassificationExample):
            inputs_a, span_locs_a = tokenize_tracking_span(tokenizer, example.text_a, example.spans_a)
            if example.spans_b is not None:
                inputs_b, span_locs_b = tokenize_tracking_span(tokenizer, example.text_b, example.spans_b)

                input_ids = inputs_a["input_ids"] + inputs_b["input_ids"][1:]
                token_type_ids = inputs_a["token_type_ids"] + ([1] * len(inputs_b["token_type_ids"][1:]))
                offset = len(inputs_a["input_ids"]) - 1
                span_locs_b = [(s + offset, e + offset) for s, e in span_locs_b]
                span_locs = span_locs_a + span_locs_b

                tmp = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
                assert tmp["input_ids"] == input_ids, "Span tracking tokenization produced inconsistent result!"
                assert tmp["token_type_ids"] == token_type_ids, "Span tracking tokenization produced inconsistent result!"
            else:
                input_ids, token_type_ids = inputs_a["input_ids"], inputs_a["token_type_ids"]
                span_locs = span_locs_a

        else:
            inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length,)
            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            # TODO(AW): will fuck up span tracking
            assert False, "Not implemented correctly wrt span tracking!"
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids

        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        if output_mode in ["classification", "span_classification"]:
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        if isinstance(example, SpanClassificationExample):
            feats = SpanClassificationFeatures(input_ids=input_ids,
                                               span_locs=span_locs,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               label=label)
        else:
            feats = InputFeatures(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  label=label)

        features.append(feats)

    if is_tf_available() and is_tf_dataset:
        # TODO(AW): include span classification version

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features


class BoolqProcessor(DataProcessor):
    """Processor for the BoolQ data set (SuperGLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line["idx"])
            text_a = line["passage"]
            text_b = line["question"]
            label = line["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CbProcessor(DataProcessor):
    """Processor for the CommitmentBank data set (SuperGLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "contradiction", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line["idx"])
            text_a = line["premise"]
            text_b = line["hypothesis"]
            label = line["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set (SuperGLUE version)."""
    # TODO(AW)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MultircProcessor(DataProcessor):
    """Processor for the Multirc data set (SuperGLUE version)."""
    # TODO(AW)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RecordProcessor(DataProcessor):
    """Processor for the ReCoRD data set (SuperGLUE version)."""
    # TODO(AW)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "dev.jsonl")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line["idx"])
            text_a = line["premise"]
            text_b = line["hypothesis"]
            label = line["label"]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WicProcessor(DataProcessor):
    """Processor for the WiC data set (SuperGLUE version)."""
    # TODO(AW)

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line["idx"])
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            span_a = (line["start1"], line["end1"])
            span_b = (line["start2"], line["end2"])
            label = line["label"]
            examples.append(SpanClassificationExample(guid=guid, text_a=text_a, spans_a=[span_a],
                                                      text_b=text_b, spans_b=[span_b], label=label))
        return examples


class WscProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_jsonl(os.path.join(data_dir, "val.jsonl")), "dev")

    def get_labels(self):
        """See base class."""
        return [True, False]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line["idx"])
            text_a = line["text"]
            span_start1 = line["target"]["span1_index"]
            span_start2 = line["target"]["span2_index"]
            span_end1 = span_start1 + len(line["target"]["span1_text"])
            span_end2 = span_start2 + len(line["target"]["span2_text"])
            span1 = (span_start1, span_end1)
            span2 = (span_start2, span_end2)
            label = line["label"]
            examples.append(SpanClassificationExample(guid=guid, text_a=text_a, spans_a=[span1, span2], label=label))
        return examples


superglue_tasks_num_labels = {
    "boolq": 2,
    "cb": 3,
    "copa": 2,
    "rte": 2,
    "wic": 2,
    "wsc": 2,
}

superglue_tasks_num_spans = {
    "wic": 2,
    "wsc": 2,
}

superglue_processors = {
    "boolq": BoolqProcessor,
    "cb": CbProcessor,
    "copa": CopaProcessor,
    "multirc": MultircProcessor,
    "record": RecordProcessor,
    "rte": RteProcessor,
    "wic": WicProcessor,
    "wsc": WscProcessor,
}

superglue_output_modes = {
    "boolq": "classification",
    "cb": "classification",
    "copa": "classification",
    "multirc": "TODO",
    "record": "TODO",
    "rte": "classification",
    "wic": "span_classification",
    "wsc": "span_classification",
}
