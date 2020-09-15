import os
import subprocess
import sys

from tqdm import tqdm

from helpers import run_cmd

# points to slp-labs/lab1/scripts
SCRIPT_DIRECTORY = os.path.realpath(__file__)


def read_test_set(fname):
    pairs = []
    with open(fname, "r") as fd:
        lines = [ln.strip().split(": ") for ln in fd.readlines()]

        for ln in lines:
            correct = ln[0]

            for wrong in ln[1].split():
                pairs.append((wrong, correct))

    return pairs


def correct_word(word, corrector):
    corrected = run_cmd(f"bash predict.sh {corrector} {word}")

    return corrected.strip()


def run_spell_checker(pairs, corrector):
    hits = 0

    for wrong, correct in tqdm(pairs):
        corrected = correct_word(wrong, corrector)
        tqdm.write(f"{wrong} -> {corrected}: {correct}")

        if corrected == correct:
            hits += 1
    print("Accuracy: {}".format(hits / len(pairs)))


if __name__ == "__main__":
    pairs = read_test_set(os.path.join(SCRIPT_DIRECTORY, "../data/spell_test.txt"))
    corrector = sys.argv[1]
    run_spell_checker(pairs, corrector)
