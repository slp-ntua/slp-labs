#!/usr/bin/env python
import argparse
import fileinput

EPS = "<epsilon>"

parser = argparse.ArgumentParser(
    prog="chcase", description="Change letter case in file"
)
parser.add_argument("-u", "--upper", action="store_true", help="Change to upper case")
parser.add_argument(
    "files",
    metavar="FILE",
    nargs="*",
    help="files to read, if empty, stdin is used",
)
args = parser.parse_args()


for ln in fileinput.input(args.files):
    word = ln.strip()
    fst_description_file = word + ".txt"
    with open(fst_description_file, "w") as fd:
        curr_idx = 0
        for i, c in enumerate(word):
            if i == 0:
                fd.write(f"{i} {i + 1} {c} {word}\n")
            else:
                fd.write(f"{i} {i+1} {c} {EPS}\n")

            curr_idx = i
        fd.write(f"{curr_idx + 1}")
