#
python3 runner.py -lt square -lc 3 -lr 2 -lb periodic -v 4 -q 12 -ql 1,2,3,5,8,11,14,13,12,10,7,4 -b ibmq_guadalupe -t None -tg compute -i 100 -m fermihubbard >> out-guadalupe-sq6-fm.txt &
python3 runner.py -lt triangular -lc 3 -lr 2 -lb periodic -v 4 -q 12 -ql 1,2,3,5,8,11,14,13,12,10,7,4 -b ibmq_guadalupe -t None -tg compute -i 100 -m fermihubbard >> out-guadalupe-tg6-fm.txt &
python3 runner.py -lt line -lc 3 -lr 2 -lb periodic -v 4 -q 12 -ql 1,2,3,5,8,11,14,13,12,10,7,4 -b ibmq_guadalupe -t None -tg compute -i 100 -m fermihubbard >> out-guadalupe-line6-fm.txt &
#python3 runner.py -lt kagome -lc 4 -lr 3 -lb periodic -v 4 -q 24 -ql 0,1,2,3,4,5,6,7,8,9,10,11 -b ibm_hanoi -t None -tg -18 -i 100 -m fermihubbard >> out-hanoi-kg12.txt &
