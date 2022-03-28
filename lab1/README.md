# LAB 1: OpenFST Spell checker and familiarization with Word2vec

## Examples
The examples folder contains the pre-lab examples we demonstrated in class.

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
├── data                            # -> Train and test corpora
│   ├── spell_test.txt              # -> spell checker evaluation corpus
│   └── wiki.txt                    # -> Wikipedial word misspellings
├── fsts                            # -> Compiled FSTs and FST description files
├── install_openfst.sh              # -> OpenFST installation script
├── README.md                       # -> This file.
├── requirements.txt                # -> Python dependencies
├── scripts                         # -> Python and Bash scripts go here
│   ├── fetch_gutenberg.py          # -> Provided script to download the gutenberg corpus
│   ├── helpers.py                  # -> Provided helper functions
│   ├── mkfstinput.py               # -> Provided script to pass a word as input to the spell checker
│   ├── predict.sh                  # -> Provided script to run prediction for a word
│   ├── run_evaluation.py           # -> Provided script to run evaluation on the test corpus
│   ├── util.py                     # -> Stubs to fill in some of your utility functions. TODO
│   ├── w2v_sentiment_analysis.py   # -> Complete the code here for sentiment analysis using word2vec
│   ├── w2v_train.py                # -> Complete the code here to train a word2vec model on the gutenberg corpus
│   └── word_edits.sh               # -> Provided script to get the minimum edit distance edits between two words
└── vocab                           # -> Place your vocab and syms files here
```
We also propose to use the `.fst` suffix for fst description files and the `.binfst` suffix for compiled fsts.

We recommend you study the code we provide before you start and try to run some basic examples.
This will give you some basic understanding about how to script for OpenFST.
Also, you will probably avoid reimplementing existing functionality


## Part 1: Spell checker using Finite state transducers

In this part you are going to implement a spell checker using Finite state transducers and the
OpenFST library. The spell checker consists of two parts: a Levenshtein transducer and a word
acceptor.

Follow the instructions in the lab handouts (available through mycourses) to complete this part.

### Spell checker evaluation

Once you have implemented a spell checker, e.g. `fsts/MY_SPELL_CHECKER.binfst` you can use the
following script for evaluation on the provided test set.

Run:

```bash
python scripts/run_evaluation.py fsts/MY_SPELL_CHECKER.binfst
```

The script will run the spell checker on the test set and print the model accuracy (percentage
of misspelled words that are corrected appropriately).


## Part 2: Familiarization with word2vec and sentiment analysis

In this part you will use the gutenberg corpus to train a word2vec model. You will find most
similar words, perform word analogies and visualize the embeddings using
[https://projector.tensorflow.org/](https://projector.tensorflow.org/). Finally you will use the
embeddings you created to perform sentiment analysis on the IMDB movie reviews dataset.

Follow the instructions in the lab handouts (available through mycourses) to complete this part.
