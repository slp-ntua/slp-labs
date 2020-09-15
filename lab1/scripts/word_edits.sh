#!/usr/bin/env bash

# Calculate the edits needed to get from a misspelled word to a correct word

# Usage:
#   bash scripts/word_edits.sh tst test
# Output:
#   <eps> e

# Command line args
WRONG=${1}
CORRECT=${2}


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

###
# Make sure to create these files
CHARSSYMS=${CURRENT_DIRECTORY}/../syms/chars.syms  # Character symbol table
VANILLA_LEVENSHTEIN=${CURRENT_DIRECTORY}/../fsts/L.binfst  # Compile basic Levenshtein FST
###

# Temp fst file. Is deleted after the script runs
MLFST=${CURRENT_DIRECTORY}/../fsts/ML.binfst


# Compose M with L to create  ML.fst
python mkfstinput.py ${WRONG} |
    fstcompile --isymbols=${CHARSSYMS} --osymbols=${CHARSSYMS} |
    fstcompose - ${VANILLA_LEVENSHTEIN} > ${MLFST}


# Create N.fst and compose ML.fst with N.fst to create MLN.fst
python mkfstinput.py ${CORRECT} |
    fstcompile --isymbols=${CHARSSYMS} --osymbols=${CHARSSYMS} |
    fstcompose ${MLFST} - |
    # Run shortest path to get the edits for the minimum edit distance
    fstshortestpath |
    # Print the shortest path fst
    fstprint --isymbols=${CHARSSYMS} --osymbols=${CHARSSYMS}  --show_weight_one |
    # Ignore the accepting state and arcs with 0 weight (no edits)
    grep -v "0$" |
    # Get columns 3 and 4 that contain source and destination symbol for the remaining columns (edits)
    cut -d$'\t' -f3-4

rm ${MLFST}
