# Make test input
fstcompile --isymbols=ascii.syms --osymbols=ascii.syms >Mars_man.fst <<EOF
0 1 M M
1 2 a a
2 3 r r
3 4 s s
4 5 <space> <space>
5 6 m m
6 7 a a
7 8 n n
8 9 ! !
9
EOF


# Compose input with lexicon.  M:<eps> -> a: <eps> -> r: <eps> -> s: Mars -> m: man -> a: <eps> -> n: <eps>
# Project isyms to osyms. <eps>: <eps> -> <eps>: <eps> -> <eps>: <eps> -> Mars: Mars -> <eps>: <eps> -> <eps>: <eps> -> man: man
fstcompose Mars_man.fst lexicon_opt.fst | fstproject --project_output | fstrmepsilon >tokens.fst
fstdraw --isymbols=wotw.syms --osymbols=wotw.syms --portrait tokens.fst | dot -Tjpg >tokens.jpg
