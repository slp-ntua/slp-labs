# LAB 1: OpenFST Spell checker and familiarization with Word2vec

## Setup

To setup openfst in your machine run.

```bash
bash install_openfst.sh
```

Leave the `OPENFST_VERSION=1.6.1`, since next versions are not supported for this lab / contain breaking changes.

Install python dependencies with:

```bash
pip install -r requirements.txt
```

Fetch the NLTK Gutenberg corpus using the following script.

```bash
python scripts/fetch_gutenberg.py > data/gutenberg.txt
```
This script downloads and preprocesses the corpus.

## Proposed code structure and provided resources

We propose the following structure to organize your code.
```
├── data                     # -> Train and test corpora
│   ├── spell\_test.txt      # -> spell checker evaluation corpus
│   └── wiki.txt             # -> Wikipedial word misspellings
├── fsts                     # -> Compiled FSTs and FST description files
├── install\_openfst.sh      # -> OpenFST installation script
├── README.md                # -> This file.
├── requirements.txt         # -> Python dependencies
├── scripts                  # -> Python and Bash scripts go here
│   ├── fetch\_gutenberg.py  # -> Provided script to download the gutenberg corpus
│   ├── helpers.py           # -> Provided helper functions
│   ├── mkfstinput.py        # -> Provided script to pass a word as input to the spell checker
│   ├── predict.sh           # -> Provided script to run prediction for a word
│   ├── run\_evaluation.py   # -> Provided script to run evaluation on the test corpus
│   ├── util.py              # -> Stubs to fill in some of your utility functions. TODO
│   └── word\_edits.sh       # -> Provided script to get the minimum edit distance edits between two words
└── vocab                    # -> Place your vocab and syms files here
```
We also propose to use the `.fst` suffix for fst description files and the `.binfst` suffix for compiled fsts.

We recommend you study the code we provide before you start and try to run some basic examples.
This will give you some basic understanding about how to script for OpenFST.
Also, you will probably avoid reimplementing existing functionality


## Evaluation

Run:

```bash
python scripts/run_evaluation.py fsts/MY_SPELL_CHECKER.binfst
```

The script will run the spell checker through the test set and print the final accuracy.
