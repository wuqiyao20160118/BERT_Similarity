# -*- coding:utf-8 -*-
"""
This module tokenize input text pairs and writes to output
"""

import sys
import json
import tensorflow as tf
import codecs
import gzip
import tokenization
import os

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None, "Plain Input file directory.")
flags.DEFINE_string("output_file", None, "Output file.")
flags.DEFINE_string("vocab_file", "pre_trained/vocab.txt", "Output file in Squad format.")
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    tokenizer = tokenization.FullTokenizer(FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for filename in os.listdir(FLAGS.input_file):
        input_files.extend(tf.io.gfile.glob(os.path.join(FLAGS.input_file, filename)))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    with codecs.open(FLAGS.output_file, "w", encoding="utf-8") as output_json:
        for input_file in input_files:
            if input_file.endswith("gz"):
                reader = gzip.open(input_file, 'rb')
            else:
                reader = tf.io.gfile.GFile(input_file, "r")

            while True:
                # label [\t] text1 [\t] text2 [\n]
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()
                line = line.split('\t')
                label = line[0]
                text1 = line[1]
                text2 = line[2]
                text1_ = " ".join(tokenizer.tokenize(text1))
                text2_ = " ".join(tokenizer.tokenize(text2))
                output_json.write(label + '\t' + text1_ + '\t' + text2_ + '\n')


if __name__ == '__main__':
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    tf.app.run()
