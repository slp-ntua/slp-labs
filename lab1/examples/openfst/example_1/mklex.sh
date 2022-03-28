# Create description files for all words
cat valid_words.txt | python create_word_fst.py

# Compile individual word fsts
fstcompile --isymbols=ascii.syms --osymbols=wotw.syms man.txt man.fst
fstcompile --isymbols=ascii.syms --osymbols=wotw.syms Mars.txt Mars.fst
fstcompile --isymbols=ascii.syms --osymbols=wotw.syms Martian.txt Martian.fst

# Compile punctuation removal fst
fstcompile --isymbols=ascii.syms --osymbols=wotw.syms full_punct.txt punct.fst

# Union of individual words + concat with punctuation remover -> lexicon
fstunion man.fst Mars.fst | fstunion - Martian.fst | fstconcat - punct.fst | fstclosure >lexicon.fst

# Optimize lexicon
fstrmepsilon lexicon.fst | fstdeterminize | fstminimize >lexicon_opt.fst
