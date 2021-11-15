import argparse
import json
import re

import nltk
from datasets import load_metric
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from dsum import Summarizer, convert_pov, parse_dialogue_string


class NLTKParser:
    def parse(self, sentence):
        sentence = word_tokenize(sentence)
        return nltk.pos_tag(sentence)

    def is_keyword_pos(self, pos_tag):
        return pos_tag in {"NN", "NNS", "NNS", "NNPS"}

    def is_verb_pos(self, pos_tag):
        return "V" in pos_tag


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Test file")
    parser.add_argument(
        "-c",
        "--convert-pov",
        action="store_true",
        default=False,
        help="Apply POV conversion module",
    )
    parser.add_argument(
        "-k", "--top-k", help="Number of summary sentences", type=int, default=2
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print summaries to console",
    )
    args = parser.parse_args()

    verbose = args.verbose

    rogue = load_metric("rouge")

    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("taggers/averaged_perceptron_tagger")
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("averaged_perceptron_tagger")

    parser = NLTKParser()
    stopwords = set(nltk.corpus.stopwords.words("english"))

    line_pattern = re.compile("#(.*)#:(.*)")

    with open(args.file, "r") as rf:
        data = json.load(rf)

    cum_r1 = cum_r2 = cum_rl = 0

    data_iterator = tqdm(data) if not verbose else data

    all_predictions = []
    all_references = []

    for d in data_iterator:

        # if d["id"] != "test_97":
        # continue

        # All dialogues have keys "summary1", "summary2", and "summary3".
        references = (d["summary"],)

        if verbose:
            print("=" * 50)
            print(f"[ORIGINAL]")

        utterances = parse_dialogue_string(d["dialogue"])
        utts = []
        for s, u in utterances:
            u_split = u.split()
            cleaned = " ".join([w for w in u_split if "<" not in w and ">" not in u])
            if cleaned:
                utts.append((s, cleaned))
        utterances = utts

        summary = " ".join(u for _, u in utterances[:3])

        # Calculate metrics.
        for i, reference in enumerate(references):
            all_predictions.append(summary)
            all_references.append(reference)

    results = rogue.compute(predictions=all_predictions, references=all_references)
    r1 = results["rouge1"].mid.fmeasure
    r2 = results["rouge2"].mid.fmeasure
    rl = results["rougeL"].mid.fmeasure

    print(
        f"[Samsum LEAD-3] R1: {r1:.4f}, R2: {r2:.4f}, Rl: {rl:.4f} (top_k: {args.top_k})"
    )
