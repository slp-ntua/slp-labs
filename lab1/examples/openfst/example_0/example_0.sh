# first of all we need to compile the fst
fstcompile -isymbols=T_0.isyms -osymbols=T_0.osyms T_0.txt T_0.fst
# Print the fst to visualize the compiled result
fstprint -isymbols=T_0.isyms -osymbols=T_0.osyms T_0.fst > T0_print.txt
# draw the FST
fstdraw --isymbols=T_0.isyms --osymbols=T_0.osyms -portrait T_0.fst | dot -Tpng >T_0.png

