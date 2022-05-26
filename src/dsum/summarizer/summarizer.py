import argparse
from functools import reduce

#import mecab
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


class Parser:
    def parse(sentence):
        pass

    def keyword_pos(tag):
        pass

    def is_verb_pos(self, pos_tag):
        pass


class MecabParser(Parser):
    def __init__(self):
        self.parser = mecab.MeCab()

    def parse(self, sentence):
        return self.parser.pos(sentence)

    def is_keyword_pos(self, pos_tag):
        return pos_tag in {"NNG", "NNP"}

    def is_verb_pos(self, pos_tag):
        return "V" in pos_tag


class Word:
    def __init__(self, word, utterance_index=None, index=None, pos=None):
        self.item = word
        self.utterance_index = utterance_index
        self.index = index
        self.pos = pos

    def __str__(self):
        return f"{self.item}|{self.utterance_index}|{self.index}|{self.pos}"


class Utterance:
    def __init__(self, index, speaker, words):
        self.index = index
        self.speaker = speaker
        self.words = words


class Node:
    def __init__(self, id, word, pos=None):
        self.id = id
        self.item = word
        self.pos = pos
        self.frequency = 0
        self.sources = {}

    def __str__(self):
        return f"{self.item}|{self.pos}"


class Summarizer:
    def __init__(self, parser, utterances, stopwords={}):
        self.parser = parser
        self.stopwords = set(stopwords)
        self.utterances = self.parse_utterances(utterances)

        self.graph = nx.DiGraph()
        self.word_to_node = {}
        self.keyword_nodes = {}
        self.speaker_keyword_nodes = {}
        self.original_paths = []
        self.keyword_count_threshold = None
        self.keyword_score_threshold = None

        # Initialize internal data links.
        self.speakers, self.speaker_nodes, self.speaker_edges = [], {}, {}
        for utterance in self.utterances:
            speaker = utterance.speaker
            if speaker not in self.speakers:
                self.speakers.append(speaker)
                self.speaker_nodes[speaker] = set()
                self.speaker_edges[speaker] = set()

        # Add initial bos and eos nodes.
        self.bos_node = self.add_new_node(Word("<bos>", None, None, "META"))
        self.eos_node = self.add_new_node(Word("<eos>", None, None, "META"))

        # Build internal word graph.
        self.build_graph()
        self.extract_keyword_nodes()
        self.find_keyword_thresholds()
        # self.break_points = self.get_topic_segments()

    def parse_utterances(self, raw_utterances):
        utterances = []
        for index, (speaker, raw_utterance) in enumerate(raw_utterances):
            raw_utterance = raw_utterance.replace("I", speaker)
            raw_utterance = raw_utterance.replace("I'm", f"{speaker} is")
            words = self.preprocess_utterance(index, raw_utterance)
            if not words:
                continue
            utterance = Utterance(speaker=speaker, index=index, words=words)
            utterances.append(utterance)
        return utterances

    def preprocess_utterance(self, index, raw_utterance):
        if not raw_utterance.strip():
            return None
        parsed = self.parser.parse(raw_utterance)
        bos_word = Word("<bos>", index, 0, "META")
        eos_word = Word("<eos>", index, len(parsed) + 1, "META")
        words = [
            Word(word.lower(), index, wi + 1, pos)
            for wi, (word, pos) in enumerate(parsed)
        ]
        return [bos_word] + words + [eos_word]

    def build_graph(self):
        for utterance in self.utterances:
            # We add nodes from a single utterance in two steps:
            # Non-stopwords first and then stopwords.
            nodeset = self.add_utterance_nodes(utterance)
            _ = self.add_utterance_nodes(utterance, is_stopword_step=True)
            self.original_paths.append(nodeset)
        for utterance in self.utterances:
            self.add_utterance_edges(utterance)

    def add_utterance_nodes(self, utterance, is_stopword_step=False):

        path = set()

        for i, word in enumerate(utterance.words):

            if is_stopword_step ^ (word.item in self.stopwords):
                # Skip if is_stopword_step = False and word is a stopword,
                # or is_stopword_step = True and word is not a stopword.
                # A XOR B (= A^B) is true iff ONE of A and B is 1.
                continue

            candidate_nodes = [
                n
                for n in self.graph.nodes
                if n.item == word.item
                and n.pos == word.pos
                and word.utterance_index not in n.sources
            ]

            # 1. No previously added node matches current word node.
            if not candidate_nodes:
                matched_node = self.add_new_node(word)

            # 2. Word is non-amiguously matches to a node.
            elif len(candidate_nodes) == 1:
                matched_node = candidate_nodes[0]

            else:  # 3. Multiple candidate nodes: ambiguous!
                matched_node = self.resolve_candidate_nodes(
                    word, candidate_nodes, is_stopword_step
                )

            self.word_to_node[word] = matched_node
            matched_node.sources[word.utterance_index] = word
            matched_node.frequency += 1

            # Update speaker's node list.
            self.speaker_nodes[utterance.speaker].add(matched_node)
            path.add(matched_node)

        return path

    def add_new_node(self, word):
        node = Node(self.graph.number_of_nodes(), word.item, word.pos)
        self.graph.add_node(node)
        return node

    def resolve_candidate_nodes(self, word, candidate_nodes, is_stopword_step):

        # Calculate candidate scores according to immediate context.
        matched_node, max_score = candidate_nodes[0], 0

        for candidate_node in candidate_nodes:

            candidate_score = self.calculate_candidate_score(word, candidate_node)

            if candidate_score > max_score:
                matched_node = candidate_node

            if candidate_score == max_score:
                # If candidate_score is a tie, choose node with more source words.

                if len(candidate_node.sources) > len(matched_node.sources):
                    matched_node = candidate_node

        if is_stopword_step and max_score <= 0:
            # Stopwords are matched only if context overlaps with another stopword.
            matched_node = self.add_new_node(word)

        return matched_node

    def calculate_candidate_score(self, word, node):

        utterance = self.utterances[word.utterance_index]
        words = utterance.words
        prev_word = words[word.index - 1] if word.index > 0 else None
        next_word = words[word.index + 1] if word.index < len(words) else None

        candidate_score = 0

        for utterance_index, source_word in node.sources.items():

            # c for context.
            words_c = self.utterances[utterance_index].words
            prev_word_c = (
                words_c[source_word.index - 1] if source_word.index > 0 else None
            )
            next_word_c = (
                words_c[source_word.index + 1]
                if source_word.index < len(words_c)
                else None
            )

            if prev_word and prev_word_c and prev_word.item == prev_word_c.item:
                candidate_score += 1

            if next_word and next_word_c and next_word.item == next_word_c.item:
                candidate_score += 1
                # candidate_score += max(1, negative_candidate_score)

        return candidate_score

    def add_utterance_edges(self, utterance):

        words = utterance.words

        if not words:
            return

        for from_word, to_word in zip(words, words[1:]):

            from_node = self.word_to_node[from_word]
            to_node = self.word_to_node[to_word]

            self.speaker_edges[utterance.speaker].add((from_node, to_node))

            if self.graph.has_edge(from_node, to_node):
                continue

            weight = self.calculate_edge_weight(from_node, to_node)
            self.graph.add_edge(from_node, to_node, weight=weight)

    def calculate_edge_weight(self, from_node, to_node):

        # Calculate weight
        combined_frequency = from_node.frequency + to_node.frequency
        word_distance = 0

        for utterance_index, from_word in from_node.sources.items():

            to_word = to_node.sources.get(utterance_index)

            if not to_word:
                continue

            diff = abs(to_word.index - from_word.index)
            # word_distance += max(0, diff)
            word_distance += diff

        if word_distance == 0:
            raise Exception("Word added twice as nodes.")

        # w1: Equation (2) in Filippova, 2010.
        # w2: Equation (4) in Filippova, 2010.
        w1 = (from_node.frequency + to_node.frequency) * word_distance
        w2 = w1 / (from_node.frequency * to_node.frequency)

        return w2

    def extract_keyword_nodes(self):
        ### KEYWORD VOTING!!!
        ## FIND GLOBAL, AS WELL AS LOCAL (PERSONALIZED) CORES!!
        ## THEN VOTE!!!!!!!!!!!!!!!!!!
        global_core = nx.algorithms.core.k_core(self.graph)
        global_core = {
            node
            for node in global_core
            if self.parser.is_keyword_pos(node.pos) and node.item not in self.stopwords
        }

        all_cores = [global_core]

        # Separate dictionaries to prevent speaker name from overwriting reserved
        # keyword dictinonary names, like "all", "union", and "intersection".
        for speaker in self.speakers:
            speaker_graph = self.get_speaker_graph(speaker)
            local_core = nx.algorithms.core.k_core(speaker_graph)
            local_core = {
                node
                for node in local_core
                if self.parser.is_keyword_pos(node.pos)
                and node.item not in self.stopwords
            }
            all_cores.append(local_core)
            self.speaker_keyword_nodes[speaker] = local_core

        self.keyword_nodes["all"] = global_core
        self.keyword_nodes["union"] = set.union(*all_cores)
        self.keyword_nodes["intersection"] = set.intersection(*all_cores)

    def find_keyword_thresholds(self, coalition_scheme="mean"):
        keyword_nodes = self.keyword_nodes["all"]
        if not keyword_nodes:
            self.keyword_count_threshold = 0
            self.keyword_score_threshold = 0
            return
        counts = [
            len(keyword_nodes.intersection(nodes)) for nodes in self.original_paths
        ]
        if coalition_scheme == "mean":
            self.keyword_count_threshold = sum(counts) / len(counts) if counts else 1
        else:
            self.keyword_count_threshold = max(counts)
        self.keyword_score_threshold = self.keyword_count_threshold / len(keyword_nodes)

    def keywords(self, key="all"):
        return [node.item for node in self.keyword_nodes[key]]

    def speaker_keywords(self, speaker):
        return [node.item for node in self.speaker_keyword_nodes[speaker]]

    def get_speaker_graph(self, speaker):

        speaker_nodes = self.speaker_nodes[speaker]
        speaker_edges = self.speaker_edges[speaker]

        speaker_graph = nx.subgraph_view(
            self.graph,
            filter_node=lambda n: n in speaker_nodes,
            filter_edge=lambda n1, n2: (n1, n2) in speaker_edges,
        )

        return speaker_graph

    def summarize(self, per_speaker=False, top_k=None):

        paths = []

        # keyword_nodes = self.keyword_nodes["intersection"]
        keyword_nodes = self.keyword_nodes["all"]

        if per_speaker:
            for speaker in self.speakers:
                speaker_graph = self.get_speaker_graph(speaker)
                candidate_paths = nx.shortest_simple_paths(
                    speaker_graph, self.bos_node, self.eos_node, weight="weight"
                )
                best_paths = list(self.rank_paths(candidate_paths, keyword_nodes))
                paths.extend(best_paths)
        else:
            candidate_paths = nx.shortest_simple_paths(
                self.graph, self.bos_node, self.eos_node, weight="weight"
            )
            paths = list(self.rank_paths(candidate_paths, keyword_nodes))

        summary, score = "", 0
        paths.sort(key=lambda x: x[0], reverse=True)

        paths = paths[:top_k] if top_k else paths
        for speaker_score, path in paths:
            # Remove <bos> and <eos>.
            speaker_summary = " ".join(node.item for node in path[1:-1])
            summary += f" {speaker_summary}"
            score += speaker_score

        normalized_score = score / len(paths) if paths else score

        return [summary], [normalized_score]

    def rank_paths(self, shortest_paths, keyword_nodes):
        # print('----')
        # print('keywords', [n.item for n in keyword_nodes])
        keyword_nodes = keyword_nodes.copy()
        keyword_count = len(keyword_nodes)
        paths_found = 0

        for i, path in enumerate(shortest_paths):
            path_score = nx.path_weight(self.graph, path, weight="weight")
            if i == 0:
                first_path = path
            if not keyword_nodes or i > 1000:
                break
            # if len(path) < 5:
            #    continue
            keywords_found = [node for node in path if node in keyword_nodes]
            has_verb = reduce(
                lambda acc, n: acc or self.parser.is_verb_pos(n.pos), path, False
            )
            score = len(keywords_found) / keyword_count
            if score >= self.keyword_score_threshold and has_verb:
                # if len(keywords_found) >= self.keyword_count_threshold and has_verb:
                for keyword_node in keywords_found:
                    keyword_nodes.discard(keyword_node)
                score = len(keywords_found) / keyword_count
                yield score, path
                paths_found += 1

        if not paths_found:
            yield 0, first_path

    def get_topic_segments(self, k):
        keyword_nodes = self.keyword_nodes["all"]
        keyword_id = {n: i for i, n in enumerate(keyword_nodes)}
        # prev_keywords_found = set()
        paths = []
        change_scores = []  # Stores tuple. (diff, index 1, index 2)
        prev_path_index, prev_keyword_vector = None, None
        # Construct keyword vector for each sequence.
        for i, path in enumerate(self.original_paths):
            keyword_found = False
            keyword_vector = np.zeros((len(keyword_nodes),))
            for node in path:
                if node in keyword_nodes:
                    keyword_vector[keyword_id[node]] = 1
                    keyword_found = True

            if keyword_found:
                if prev_path_index is not None:
                    score = np.dot(keyword_vector, prev_keyword_vector)
                    # score = (keyword_vector - prev_keyword_vector)
                    # score[score < 0] = 0
                    # score = score.sum()
                    change_score = (score, prev_path_index, i)
                    change_scores.append(change_score)

                prev_path_index, prev_keyword_vector = i, keyword_vector

        change_scores.sort(key=lambda x: x[0], reverse=True)
        # print(len(change_scores))

        # if len(change_scores) < k:
        #    chunksize = len(self.original_paths) // k
        # print('-- RANDO')
        # change_points =  [i for i in range(0, len(self.original_paths), chunksize)]

        # else:
        change_points = [0]
        # for score, index1, index2 in change_scores[: k - 1]:
        for _, index1, index2 in change_scores[: k - 1]:
            # change_point = (index2 + index1) // 2
            change_point = index2
            if change_point:
                change_points.append(change_point)

        if len(change_points) < k / 3:
            # We revert back to random splits if change_points is ridiculously low.
            chunksize = len(self.original_paths) // k
            change_points = [i for i in range(0, len(self.original_paths), chunksize)]

        change_points.sort()

        if change_points[-1] != len(self.original_paths) - 1:
            change_points.append(len(self.original_paths) - 1)
        return change_points

    def json_graph(self, speaker=None):
        graph = self.get_speaker_graph(speaker) if speaker else self.graph
        return json_graph.node_link_data(graph)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--text", help="Text to summarize")
    parser.add_argument("-f", "--file", help="Text file to summarize")
    args = parser.parse_args()

    pos_parser = MecabParser()
    stopwords_file = "src/dsum/word_graph/stopwords.txt"

    with open(args.file, "r") as rf:
        sentences = [line.strip() for line in rf]

    with open(stopwords_file, "r") as rf:
        stopwords = {line.strip() for line in rf}

    s = Summarizer(parser=pos_parser, stopwords=stopwords, sentences=sentences)

    import matplotlib
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    from matplotlib import rc

    plt.rcParams["font.sans-serif"] = ["NanumBarunGothic"]
    plt.rcParams["font.family"] = ["NanumBarunGothic"]
    plt.figure(figsize=(8, 8))

    pos = nx.spring_layout(s.graph, scale=0.2)

    node_colors = []
    for node in s.graph.nodes:
        if node.pos == "META":
            node_colors.append("#ee7878")
        elif node.item in s.stopwords:
            node_colors.append("#C6D8D3")
        else:
            node_colors.append("#58A4B0")

    nx.draw_networkx_nodes(
        s.graph,
        pos=pos,
        node_color=node_colors,
    )
    nx.draw_networkx_edges(s.graph, pos=pos, arrowsize=10)
    y_off = 0.01
    nx.draw_networkx_labels(
        s.graph, pos={k: ([v[0], v[1] + y_off]) for k, v in pos.items()}
    )
    edge_labels = nx.get_edge_attributes(s.graph, "weight")
    nx.draw_networkx_edge_labels(s.graph, pos=pos, edge_labels=edge_labels)
    plt.savefig("graph.png")

    print("KEYWORDS:", {n.item for n in s.keyword_nodes()})

    summaries, scores = s.summarize(top_k=10)

    print("BEST SUMMARY:", summaries[0])
    print("CANDIDATE SUMMARIES: ----")
    for summary, path_score in zip(summaries, scores):
        print(f"- [{path_score}] {summary}")
