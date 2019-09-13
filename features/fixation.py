import numpy as np
import pandas as pd
import csv


for i in range(1, 7):
    subject = i
    numberOfLines = 11
    story = 6

    df = pd.read_excel("data/Test6Subject"+str(subject)+".xlsx")

    gazeEvent = df[["GazeEventType", "GazeEventDuration"]].values
    RecordingTimestamp = df["RecordingTimestamp"].values

    # * Details About The Page
    pageStart = [i for i, x in enumerate(
        df["StudioEvent"].values) if(x == "PDFPageStarted")]
    pageEnd = [i for i, x in enumerate(
        df["StudioEvent"].values) if(x == "PDFPageEnded")]
    numberOfPages = len(pageStart)

    # * Total Recording Time
    totalTime = 0
    for i in range(1):
        totalTime += RecordingTimestamp[pageEnd[i]] - \
            RecordingTimestamp[pageStart[i]]

    # * Fixation Features
    fixations = np.array([])
    for i in range(1):
        j = pageStart[i]
        while(j < pageEnd[i]):
            if(gazeEvent[j][0] == "Fixation"):
                fixations = np.append(fixations, gazeEvent[j][1])
                while(gazeEvent[j+1][0] == "Fixation" and gazeEvent[j+1][1] == gazeEvent[j][1]):
                    j += 1
            j += 1

    fixationCount = len(fixations)
    totalFixationDuration = fixations.sum()
    minFixationDuration = fixations.min()
    maxFixationDuration = fixations.max()
    averageFixationDuration = fixations.mean()
    fixationFrequency = fixationCount/(totalTime/1000)


# print(fixationCount, totalFixationDuration,
#       averageFixationDuration, fixationFrequency)

    row = [story, subject, fixationCount, totalFixationDuration, averageFixationDuration,
           fixationFrequency, fixationCount/11, totalFixationDuration/11]

    with open('check.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
