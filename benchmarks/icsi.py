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
    parser.add_argument("-d", "--data-dir", help="Directory with test files")
    parser.add_argument(
        "-c",
        "--convert-pov",
        action="store_true",
        default=False,
        help="Apply POV conversion module",
    )
    parser.add_argument(
        "-l", "--min-length", help="Minimum utt length", type=int, default=45
    )
    parser.add_argument(
        "-t", "--topic-groups", help="Topic groups", type=int, default=25
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
    test_dir = Path(args.data_dir) / Path("test")
    reference_dir = Path(args.data_dir) / Path("reference")
    all_references = defaultdict(list)

    for reference_file in reference_dir.glob("**/*.txt"):
        with open(reference_file, "r") as rf:
            references = rf.readlines()
            all_references["Bed004"].append(references[0].strip())
            all_references["Bed009"].append(references[1].strip())
            all_references["Bed016"].append(references[2].strip())
            all_references["Bmr005"].append(references[3].strip())
            all_references["Bmr019"].append(references[4].strip())
            all_references["Bro018"].append(references[5].strip())

    for meeting_file in test_dir.glob("**/*.txt"):

        meeting_name = meeting_file.stem

        transcript = ""
        with open(meeting_file, "r") as rf:
            for line in rf:
                [speaker, utterance] = line.split("\t")
                transcript += f"{speaker}:{utterance}\n"

        data.append(
            {"dialogue": transcript[:-1], "summary": all_references[meeting_name]}
        )
        # data = [json.loads(line) for line in rf]

    cum_r1 = cum_r2 = cum_rl = 0

    data_iterator = tqdm(data) if not verbose else data

    for d in data_iterator:

        # All dialogues have keys "summary1", "summary2", and "summary3".
        references = d["summary"]

        if verbose:
            print("=" * 50)
            print(f"[ORIGINAL]")

        utterances = parse_dialogue_string(d["dialogue"])
        utterances = [u for u in utterances if len(u[1]) > int(args.min_length)]

        if verbose:
            for speaker, utterance in utterances:
                # print(f"    - {speaker}: {utterance}")
                pass

        summary = ""
        summary_length = 1  # 6
        chunk = len(utterances) // summary_length

        summarizer = Summarizer(
            parser=parser, utterances=utterances, stopwords=stopwords
        )
        # break_points = summarizer.get_topic_segments(5)

        # break_points = break_points[::k]
        break_points = summarizer.get_topic_segments(args.topic_groups)
        # if len(break_points) < k:
        #      break_points = [i for i in range(0, len(utterances), k)]
        #  else:
        #      break_points = break_points[::k]

        # for i in range(0, len(utterances), chunk):
        for start, end in zip(break_points, break_points[1:]):
            # topic_group = utterances[i : i + chunk]
            topic_group = utterances[start:end]
            # Summarize.
            summarizer = Summarizer(
                parser=parser, utterances=topic_group, stopwords=stopwords
            )
            summaries, _ = summarizer.summarize(per_speaker=True, top_k=args.top_k)
            s = summaries[0]
            s = s.replace("_", "")
            s = s.replace(" 'nt", "'nt")
            s = s.replace(" 's", "'s")
            s = s.replace(" 've", "'ve")
            s = s.replace(" 're", "'re")
            s = s.replace(" 'm", "'m")
            s = s.replace(", ,", ",")
            if s and args.convert_pov:
                s = convert_pov(s)
            summary += s

        # summary = summary[:len(summary)]
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
        # max_rouge1, max_rouge2, max_rougel = 0, 0, 0
        r1, r2, rl = 0, 0, 0
        for i, reference in enumerate(references):
            results = rogue.compute(predictions=[summary], references=[reference])
            if verbose:
                print(f"[REFERENCE {i}] {reference}")
            # max_rouge1 = max(max_rouge1, results["rouge1"].mid.fmeasure)
            # max_rouge2 = max(max_rouge2, results["rouge2"].mid.fmeasure)
            # max_rougel = max(max_rougel, results["rougeL"].mid.fmeasure)

            r1 += results["rouge1"].mid.fmeasure
            r2 += results["rouge2"].mid.fmeasure
            rl += results["rougeL"].mid.fmeasure

        avg_r1 = r1 / len(references)
        avg_r2 = r2 / len(references)
        avg_rl = rl / len(references)

        if verbose:
            print(f"[ICSI] R1: {avg_r1:.4f}, R2: {avg_r2:.4f}, Rl: {avg_rl:.4f}")

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
        f"[ICSI Final] R1: {final_r1:.4f}, R2: {final_r2:.4f}, Rl: {final_rl:.4f} (topic_groups: {args.topic_groups}, top_k: {args.top_k}, min_length: {args.min_length})"
    )
