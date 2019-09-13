import numpy as np
import pandas as pd
import csv

for i in range(1, 7):
    subject = i
    numberOfLines = 42
    story = 6

    df = pd.read_excel("data/Test6Subject"+str(subject)+".xlsx")

    gazeEvent = df["GazeEventType"].values
    RecordingTimestamp = df["RecordingTimestamp"].values
    fX = df["FixationPointX (MCSpx)"].values
    fY = df["FixationPointY (MCSpx)"].values

    # * Details About The Page
    pageStart = [i for i, x in enumerate(
        df["StudioEvent"].values) if(x == "PDFPageStarted")]
    pageEnd = [i for i, x in enumerate(
        df["StudioEvent"].values) if(x == "PDFPageEnded")]
    numberOfPages = len(pageStart)

    fixationsX = np.array([])
    fixationsY = np.array([])
    time = np.array([])
    # * Recording Saccadic Features
    for i in range(numberOfPages):
        j = pageStart[i]
        while(j < pageEnd[i]):
            if(gazeEvent[j] == "Fixation" and not(np.isnan(fX[j]))):
                fixationsX = np.append(fixationsX, fX[j])
                fixationsY = np.append(fixationsY, fY[j])
                time = np.append(time, RecordingTimestamp[j])
            while(j < pageEnd[i] and (((fX[j+1] == fX[j]) and (fY[j+1] == fY[j])) or np.isnan(fX[j+1]))):
                j += 1
            # if(j == pageEnd[i]):
            #     break
            j += 1

    ampX = np.diff(fixationsX)
    ampY = np.diff(fixationsY)
    timeDiff = np.diff(time)
    saccades = np.array(np.sqrt(ampX*ampX+ampY*ampY))

    regressionTime = np.array([])

    # TODO: Regression Features
    for i in range(len(ampX)):
        if(ampX[i] < 0 and abs(ampY[i]) < 30):
            regressionTime = np.append(regressionTime, timeDiff[i])

    numberOfRegressions = len(regressionTime)
    totalRegressionDuration = regressionTime.sum()
    averageRegressionDuration = regressionTime.mean()
    maxRegressionDuration = regressionTime.max()
    minRegressionDuration = regressionTime.min()

    # print(numberOfRegressions, totalRegressionDuration, averageRegressionDuration)

    row = [story, subject, numberOfRegressions, totalRegressionDuration,
           averageRegressionDuration, numberOfRegressions/42, totalRegressionDuration/42]

    with open('newData/regressions.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
