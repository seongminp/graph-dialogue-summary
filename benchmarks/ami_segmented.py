import argparse
import json
import re
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
        "-s", "--segments-dir", help="Directory with segmented transcripts"
    )
    parser.add_argument("-m", "--meeting-names", help="Name of meetings to test")
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
        "-k", "--top_k", help="Number of summary sentences", type=int, default=1
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
    segments_dir = Path(args.segments_dir) / Path("test")
    reference_dir = Path(args.data_dir) / Path("final_summary")
    for meeting_file in segments_dir.glob("**/*.txt"):
        # for meeting_name in args.meeting_names.split(","):
        # meeting_file = Path(args.data_dir) / Path(f"{meeting_name}.json")
        meeting_file_str = meeting_file.name
        meeting_name = meeting_file_str[: meeting_file_str.find("_")]
        reference_file = reference_dir / Path(meeting_name + ".txt")

        # Load utterance segment transcript.
        # with open(Path(args.segments_dir) / Path(f"{meeting_id}_comms.txt"), "r") as rf:
        with open(meeting_file, "r") as rf:
            raw_segments = rf.read().split("\n\n")

        segments = []
        for raw_segment in raw_segments:
            segment = ""
            for raw_utterance in raw_segment.split("\n"):
                if len(raw_utterance) > int(args.min_length):
                    segment += f"A:{raw_utterance.strip()}\n"
                # segment += f"A:{raw_utterance.strip()}\n"
            segments.append(segment[:-1])  # Remove trailing newline.

        with open(reference_file) as rf:
            reference = " ".join(line.strip() for line in rf)

        data.append({"segments": segments, "summary": reference})
        # data = [json.loads(line) for line in rf]

    cum_r1 = cum_r2 = cum_rl = 0

    data_iterator = tqdm(data) if not verbose else data

    for d in data_iterator:

        # All dialogues have keys "summary1", "summary2", and "summary3".
        references = [d["summary"]]

        if verbose:
            print("=" * 50)
            print(f"[ORIGINAL]")

        segments = [parse_dialogue_string(segment) for segment in d["segments"]]
        # utterances = [u for u in utterances if len(u[1]) > 20]

        if verbose:
            # for speaker, utterance in utterances:
            # print(f"    - {speaker}: {utterance}")
            pass

        candidates = []
        for segment in segments:
            # Summarize.
            if len(segment) == 0:
                continue
            summarizer = Summarizer(
                parser=parser, utterances=segment, stopwords=stopwords
            )
            summaries, scores = summarizer.summarize(top_k=args.top_k)
            s = summaries[0]
            s = s.replace("_", "")
            s = s.replace(" 'nt", "'nt")
            s = s.replace(" 's", "'s")
            s = s.replace(" 've", "'ve")
            s = s.replace(" 're", "'re")
            s = s.replace(" 'm", "'m")
            s = s.replace(", ,", ",")
            if s.strip() and args.convert_pov:
                s = convert_pov(s)
            candidates.append((s, scores[0]))

        summary = " ".join(c[0] for c in candidates)

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
        r1 = r2 = rl = 0
        for i, reference in enumerate(references):
            reference = reference.lower()
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
            print(
                f"[AMI Segmented] R1: {avg_r1:.4f}, R2: {avg_r2:.4f}, Rl: {avg_rl:.4f}"
            )

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
        f"[AMI Segmented Final] R1: {final_r1:.4f}, R2: {final_r2:.4f}, Rl: {final_rl:.4f} (top_k: {args.top_k}, min_length: {args.min_length})"
    )
