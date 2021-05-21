import stanza
import codecs
import argparse
import os


def extract_nouns(en_nlp, text):
    doc = en_nlp(text)
    nouns = set()
    for sent in doc.sentences:
        candidate = ""
        for word in sent.words:
            if word.deprel == "nsubj":
                candidate += word.text if len(candidate) == 0 else " " + word.text
            else:
                if len(candidate) > 0:
                    nouns.add(candidate)
                    candidate = ""
    return nouns


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""

    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def main(_):
    en_nlp = stanza.Pipeline('en', use_gpu=True)

    input_files = []
    for filename in os.listdir(args.input_file):
        input_files.extend(os.path.join(args.input_file, filename))

    nouns = set()
    with codecs.open(args.output_file, "w", encoding="utf-8") as out:
        for input_file in input_files:
            reader = open(input_file, "r")

            while True:
                # label [\t] text1 [\t] text2 [\n]
                line = convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()
                line = line.split('\t')
                text1, text2 = line[1], line[2]
                noun_set1, noun_set2 = extract_nouns(en_nlp, text1), extract_nouns(en_nlp, text2)
                nouns = nouns.union(noun_set1.union(noun_set2))
            print("  File %s stanza extraction done!!!", input_file)

        for kw in nouns:
            print(kw, file=out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT Similarity Project')
    parser.add_argument('--input_file', type=str, required=True, help='Plain Input file directory.')
    parser.add_argument('--input_file', type=str, required=True, help='Output file.')
    args = parser.parse_args()
    main(args)
