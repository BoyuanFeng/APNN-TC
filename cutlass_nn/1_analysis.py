#!/usr/bin/env python3
import re
import sys 

if len(sys.argv) < 2:
    raise ValueError("Usage: ./1_analysis.py result.log")

fp = open(sys.argv[1], "r")

dataset_li = []
time_li = []
for line in fp:
    if "(ms):" in line:
        time = re.findall(r'[0-9].[0-9]+', line)[0]
        print(time)
        time_li.append(float(time))
fp.close()
print("{} (ms): {:.3f}".format(sys.argv[1].strip(".log"), sum(time_li)))
# fout = open(sys.argv[1].strip(".log")+".csv", 'w')
# fout.write("dataset,Avg.Epoch (ms)\n")
# for data, time in zip(dataset_li, time_li):
#     fout.write("{},{}\n".format(data, time))
# fout.close()