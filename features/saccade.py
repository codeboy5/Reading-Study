import numpy as np
import pandas as pd
import csv

for i in range(6, 7):
    subject = i
    numberOfLines = 42
    story = 3

    df = pd.read_excel("data/Test3Subject"+str(subject)+".xlsx")

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
            j += 1

    ampX = np.diff(fixationsX)
    ampY = np.diff(fixationsY)
    timeDiff = np.diff(time)
    saccades = np.array(np.sqrt(ampX*ampX+ampY*ampY))

    #! Saccades
    saccadeLength = saccades.sum()
    saccadeCount = len(saccades)
    # saccadeFrequency = saccadeCount

    #! Saccade Duration
    totalSaccadeDuration = timeDiff.sum()
    averageSaccadeDuration = timeDiff.mean()
    maxSaccadeDuration = timeDiff.max()
    minSaccadeDuration = timeDiff.min()

    saccadesVelocity = saccades/timeDiff

    print(saccadesVelocity.mean()*1000)

    # row = [story, subject, saccadeLength, saccadeCount, averageSaccadeDuration]
    # with open('newData/saccades.csv', 'a') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerow(row)
