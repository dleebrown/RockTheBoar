# takes in a run length encoded csv of standard form, removes any entries corresponding to pix_threshold pixels or less
# writes to a new csv in order to preserve the original file since they take so long to generate

import csv
import matplotlib.pyplot as plt
import numpy as np

# control whether to plot histogram of run lengths
histogram = False

pix_threshold = 30

input_csv = '/home/donald/Desktop/test_out.csv'
output_csv = '/home/donald/Desktop/test_out_trim.csv'

inputfile = open(input_csv, mode='r')
outputfile = open(output_csv, mode='w')

writer = csv.writer(outputfile, delimiter=',')

counter = 0
for line in inputfile:
    if counter == 0:
        splitit = line.split(',')
        splitit = [splitit[i].strip() for i in range(len(splitit))]
        writer.writerow(splitit)
        counter += 1
    else:
        imname, rle = line.split(',')[0], line.split(',')[1]
        rle = rle.split()
        new_rle = ''

        buildhistogram = []

        for i in range(1, len(rle), 2):
            if histogram:
                buildhistogram.append(int(rle[i]))
            if int(rle[i]) >= pix_threshold:
                new_rle += rle[i-1]+' '
                new_rle += rle[i]+' '
        writeline = [imname, new_rle]
        writer.writerow(writeline)
        if histogram:
            n, bins, patches = plt.hist(buildhistogram, max(buildhistogram), normed=1, facecolor='FireBrick')
            plt.xlim([0, 25])
            plt.show()

inputfile.close()
outputfile.close()
