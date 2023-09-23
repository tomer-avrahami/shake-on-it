
import os
from datetime import datetime

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
import pandas as pd

# requestedFields = [""]

CSV_FILE_PREFIX = "merged_csi_ctrl"
outputDirPath = '../../data/db_files'
inputDirPath = './outputs'

csvFileName = "{}_{}.csv".format(CSV_FILE_PREFIX, timestamp_str)
csvPath = os.path.join(outputDirPath, csvFileName)
# csvLog = open(csvPath, 'w', newline='')

out_data = pd.DataFrame()
list_ = []
out_data = pd.DataFrame()

for i in os.listdir(inputDirPath):
    filePath = os.path.join(inputDirPath, i)
    if os.path.isfile(filePath):
        with open(filePath, 'r') as inputFile:
            print("start importing file " + i)
            try:
                df = pd.read_csv(filePath)
                # list_.append(df)
                out_data = pd.concat([out_data, df])
            except Exception as e:
                print(f"Failed to import file {i} error {e}")

out_data.to_csv(csvPath)

print("Done")