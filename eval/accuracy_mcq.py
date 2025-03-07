import os
import argparse
import json
from typing import List, Dict, Optional, Union
import re
import numpy as np


AREAS = ["semantics", "memory", "abstraction", "physics"]

SUB_TAGS = [
    "motion",
    "place recognition",
    "action counting",
    "spatial relations",
    "collision",
    "sequencing",
    "object recognition",
    "language",
    "action recognition",
    "distractor action",
    "distractor object",
    "state recognition",
    "task completion",
    "object counting",
    "change detection",
    "adversarial action",
    "object attributes",
    "pattern breaking",
    "feature matching",
    "pattern discovery",
    "part recognition",
    "material",
    "event recall",
    "containment",
    "occlusion",
    "solidity",
    "object permanence",
    "colour recognition",
    "conservation",
    "quantity",
    "event counting",
    "visual discrimination",
    "general knowledge",
    "stability",
]


def extract_characters_regex(s, option_regex):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
        "Answer:",
        "Option:",
        "The correct answer",
        "The correct option",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search(f"[{option_regex}]", s):
        return ""
    matches = re.search(rf"[{option_regex}]", s)
    if matches is None:
        return ""
    return matches[0]


class IncrementalMCQAcc:

    def __init__(self, dataset):
        self.total = {"correct": 0, "answered": 0}
        self.option_regex = "12345" if dataset is "charades" else "ABC"

        self.q_areas = {}
        for area in AREAS:
            self.q_areas[area] = {"correct": 0, "answered": 0}
        self.q_tags = {}
        for tag in SUB_TAGS:
            self.q_tags[tag] = {"correct": 0, "answered": 0}

    def eval_results(self, gt_answer, your_answer, q_area, q_tag):

        response = your_answer

        # Extract the answer from the response
        extration = extract_characters_regex(response, self.option_regex)

        if extration != "":
            self.q_areas[q_area]["answered"] += 1
            self.q_areas[q_area]["correct"] += extration == gt_answer
            for tag in q_tag:
                self.q_tags[tag]["answered"] += 1
                self.q_tags[tag]["correct"] += extration == gt_answer

    def print_areas_acc(self):
        print("=========Area Evaluation=========")
        for area, data in self.q_areas.items():
            print(
                f"Accuracy in arae {area} is: {100 * data['correct'] / data['answered'] if data['answered'] > 0 else 0 : .1f}% \
                ({data['correct']}/{data['answered']})"
            )

    def print_tags_acc(self):
        print("=========Tags Evaluation=========")
        for tag, data in self.q_tags.items():
            print(
                f"Accuracy in arae {tag} is: {100 * data['correct'] / data['answered'] if data['answered'] > 0 else 0 : .1f}% \
                ({data['correct']}/{data['answered']})"
            )

    def print_total_acc(self):
        print("=========Tatol Evaluation=========")
        print(
            f"Accuracy in total questions is: {100 * self.total['correct'] / self.total['answered'] if self.total['answered'] > 0 else 0 : .1f}% \
            ({self.total['correct']} / {self.total['answered']})"
        )
