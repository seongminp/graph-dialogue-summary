import argparse
import re

import nltk


def parse_dialogue_string(dialog_string):

    line_pattern = re.compile("([^:]*):(.*)")

    utterances = []
    for line in dialog_string.split("\n"):

        if not line.strip():
            continue

        line_match = line_pattern.match(line)

        if line_match:
            speaker = line_match.group(1).strip()
            utterance = line_match.group(2).strip()
        else:
            raise Exception("Failed to parse dialogue!")

        segmented_utterances = nltk.sent_tokenize(utterance)
        speaker_utterances = [(speaker, u) for u in segmented_utterances]
        utterances.extend(speaker_utterances)

    return utterances


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--argument", help="Example argument.")
    args = parser.parse_args()
