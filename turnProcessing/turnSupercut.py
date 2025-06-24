import pandas as pd
from goProPumpingParser import create_supercut, list_and_sort_files
from datetime import datetime
import numpy as np
import gpmf
import matplotlib.pyplot as plt

input_csv = "output_turn_timestamp.csv"
params_txt = "turn_params.txt"

def main():
    input_Vids_dict = list_and_sort_files("./")

    keys = list(input_Vids_dict.keys())
    if not keys:
        print("No GoPro videos found in the current directory.")
        return

    streams = []
    streams.append(gpmf.io.extract_gpmf_stream(input_Vids_dict[keys[0]][0]))

    stream = b''.join(streams)

    utcTimes = list(map(lambda x: datetime.fromisoformat(x.value).timestamp(),gpmf.parse.filter_klv(stream,"GPSU")))

    index_of_first_accurate_timestamp=0
    for i in range(len(utcTimes)-1):
        if utcTimes[i+1]-utcTimes[1]>100000:
            index_of_first_accurate_timestamp=i+1
            break

    #utcTimes = utcTimes[index_of_first_accurate_timestamp:]

    firstTimeStamp=utcTimes[index_of_first_accurate_timestamp]-index_of_first_accurate_timestamp

    utcTimes = utcTimes[index_of_first_accurate_timestamp:]

    plt.plot(utcTimes)
    plt.show()

    df = pd.read_csv(input_csv)

    df = df.rename(columns={'start_time': 'start_str', 'end_time': 'end_str'})

    def formatTime(time_str):
        print(time_str)
        print(firstTimeStamp)
        time = (float(time_str)-firstTimeStamp)
        print(time)
        print(datetime.fromtimestamp(time_str))
        if time<0:
            raise Exception("Time is negative")
        return datetime.fromtimestamp(time).strftime("%H:%M:%S")

    df['start_str'] = df['start_str'].apply(formatTime)
    df['end_str'] = df['end_str'].apply(formatTime)

    with open(params_txt, 'r') as f:
        paramList = f.read().strip().split('\n')
    params = {}
    for param in paramList:
        key, value = param.split('=')
        params[key.strip()] = value.strip()
    
    
    if params["Turn Numbers"] == "all":
        df = df[params['Type']==df['turn_type']]
    else:
        params['Turn Number'] = np.uint32(params['Turn Numbers'].split(","))
        df = df[params['Turn Number']]



    for key in input_Vids_dict:
        create_supercut(input_Vids_dict[key],df)





if __name__== "__main__":
    main()