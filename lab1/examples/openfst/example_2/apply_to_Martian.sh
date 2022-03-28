fstproject Martian.fst tmp_martian.fst 
fstcompose tmp_martian.fst full_downcase.fst tmp_martian2.fst
fstproject --project_output tmp_martian2.fst martian.fst
