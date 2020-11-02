#!/usr/bin/env bash

# Run spell corrector for an input word

# Usage:
#   bash scripts/predict.sh MY_SPELL_CHECKER tst
# Output:
#   test

# Command line args
SPELL_CHECKER_COMPILED=${1}
WORD=${2}


# Constants.
CURRENT_DIRECTORY=$(dirname $0)

###
# Make sure these files exist
CHARSYMS=${CURRENT_DIRECTORY}/../syms/chars.syms
WORDSYMS=${CURRENT_DIRECTORY}/../syms/word.syms
###

# Make input fst for the misspelled word
python mkfstinput.py ${WORD} |
    # Compile and compose with the spell checker
    fstcompile --isymbols=${CHARSYMS} --osymbols=${CHARSYMS} |
    fstcompose - ${SPELL_CHECKER_COMPILED} |
    # Get shortest path and sort arcs
    fstshortestpath |
    fstrmepsilon |
    fsttopsort |
    # print output fst using words.syms
    fstprint -osymbols=${WORDSYMS} |
    # Get destination word (corrected)
    cut -f4 |
    # Ignore epsilon outputs
    grep -v "<eps>" |
    # Ignore accepting state line
    head -n -1 |
    # Remove trailing new line
    tr -d '\n'
