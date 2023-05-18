#
python3 runner.py -lt square -lc 4 -lr 4 -lb periodic -v 4 -q 16 -ql 1,2,3,5,8,11,14,13,12,10,7,4,0,9,6,15 -b ibmq_guadalupe -t ibmq_guadalupe -tg compute -i 100 >> out-guadalupe-sq-heis.txt &
python3 runner.py -lt triangular -lc 4 -lr 4 -lb periodic -v 4 -q 16 -ql 1,2,3,5,8,11,14,13,12,10,7,4,0,9,6,15 -b ibmq_guadalupe -t ibmq_guadalupe -tg compute -i 100 >> out-guadalupe-tg-heis.txt &
python3 runner.py -lt line -lc 4 -lr 4 -lb periodic -v 4 -q 16 -ql 1,2,3,5,8,11,14,13,12,10,7,4,0,9,6,15 -b ibmq_guadalupe -t ibmq_guadalupe -tg compute -i 100 >> out-guadalupe-line-heis.txt &
python3 runner.py -lt kagome -lc 4 -lr 4 -lb periodic -v 4 -q 16 -ql 1,2,3,5,8,11,14,13,12,10,7,4,0,9,6,15 -b ibmq_guadalupe -t ibmq_guadalupe -tg compute -i 100 >> out-guadalupe-kg-heis.txt &
