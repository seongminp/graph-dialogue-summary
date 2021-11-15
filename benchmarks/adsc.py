import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

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
    parser.add_argument("-f", "--data-file", help="JSON file with test data")
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

    data = []
    with open(args.data_file, "r") as rf:
        data = json.load(rf)

    cum_r1 = cum_r2 = cum_rl = 0

    data_iterator = tqdm(data) if not verbose else data

    for d in data_iterator:

        if verbose:
            print("=" * 50)
            print(f"[ORIGINAL]")

        utterances = parse_dialogue_string(d["dialogue"])
        # utterances = [u for u in utterances if len(u[1]) > 50]

        if verbose:
            for speaker, utterance in utterances:
                print(f"    - {speaker}: {utterance}")

        summarizer = Summarizer(
            parser=parser, utterances=utterances, stopwords=stopwords
        )
        summaries, scores = summarizer.summarize(per_speaker=True, top_k=args.top_k)
        s = summaries[0]
        s = s.replace("_", "")
        s = s.replace(" 'nt", "'nt")
        s = s.replace(" 's", "'s")
        s = s.replace(" 've", "'ve")
        s = s.replace(" 'm", "'m")
        s = s.replace(", ,", ",")
        if s and args.convert_pov:
            s = convert_pov(s)
        summary = s

        # Find common keywords.
        if verbose:
            print(f"[KEYWORDS] {summarizer.keywords()}")
            print(f"    - Union KEYWORDS:  {summarizer.keywords('union')}")
            print(f"    - All KEYWORDS:  {summarizer.keywords('all')}")
            for speaker in summarizer.speakers:
                print(
                    f"    - {speaker} KEYWORDS:  {summarizer.speaker_keywords(speaker)}"
                )

        if verbose:
            print(f"[SUMMARY] {summary}")

        # Calculate metrics.
        references = d["summaries"]
        r1, r2, rl = 0, 0, 0
        for i, reference in enumerate(references):
            results = rogue.compute(predictions=[summary], references=[reference])
            if verbose:
                print(f"[REFERENCE {i}] {reference}")
            r1 += results["rouge1"].mid.fmeasure
            r2 += results["rouge2"].mid.fmeasure
            rl += results["rougeL"].mid.fmeasure

        avg_r1 = r1 / len(references)
        avg_r2 = r2 / len(references)
        avg_rl = rl / len(references)

        if verbose:
            print(f"[ADSC] R1: {avg_r1:.4f}, R2: {avg_r2:.4f}, Rl: {avg_rl:.4f}")

        cum_r1 += avg_r1
        cum_r2 += avg_r2
        cum_rl += avg_rl

    data_size = len(data)

    final_r1, final_r2, final_rl = (
        cum_r1 / data_size,
        cum_r2 / data_size,
        cum_rl / data_size,
    )
    print(
        f"[ADSC Final] R1: {final_r1:.4f}, R2: {final_r2:.4f}, Rl: {final_rl:.4f} (top_k: {args.top_k})"
    )
