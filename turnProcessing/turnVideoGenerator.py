import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import gpmf
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os
import ffmpeg


def hhmmss_to_seconds(hhmmss):
    t = datetime.strptime(hhmmss, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

def list_and_sort_files(directory):
    pattern = re.compile(r"G.(\d{2})(\d{4})\..+")
    files_dict = defaultdict(list)
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            xxx, yyy = match.groups()
            files_dict[yyy].append((int(xxx), filename))
    
    # Sort each list by XXX

    for key in files_dict:
        files_dict[key].sort()
        files_dict[key] = [file[1] for file in files_dict[key]]
    
    
    return dict(files_dict)
    

def create_supercut(video_urls, timestamp_pairs, time_offset=0 , args={}):
    total_offset = 0  # Keeps track of cumulative video duration
    # try:
    for i, video_url in enumerate(video_urls):
        # Get video duration
        # Create a directory for the video name if it doesn't exist
        print("processing video: ", video_url)
        video_name = os.path.splitext(os.path.basename(video_url))[0]
        video_dir = os.path.join(os.getcwd(), video_name[0:2]+video_name[4:])
        os.makedirs(video_dir, exist_ok=True)



        probe = ffmpeg.probe(video_url)
        duration = float(next(stream for stream in probe["streams"] if stream["codec_type"] == "video")["duration"])
        
        for j, timestamps in enumerate(timestamp_pairs):
            
            start_time = hhmmss_to_seconds(timestamps["start_str"]) - total_offset + time_offset

            end_time = min(hhmmss_to_seconds(timestamps["end_str"]) - total_offset + time_offset,duration)

            if start_time < 0 or start_time > duration or end_time < 0:
                continue
                        

            if args.get("add_10s"):
                if start_time > 10:
                    start_time = start_time - 10
                    timestamps["start"]-= timedelta(seconds=10)
                else:
                    start_time = 0
                    timestamps["start"] = timestamps["start"] - timedelta(seconds=start_time)

                end_time = end_time + 10
                
            if end_time > 0:  # Only process if clip falls within this video
                temp_file = os.path.join(video_dir, f"{video_name[0:2]+video_name[4:]}_{timestamps["turnIndex"]}.mp4")

                if args.get("add_timestamp"):
                    # Create timestamp text overlay for each timestamp in the vector
                    # Construct drawtext with pts:localtime 


                    if hasattr(timestamps["start"], "to_pydatetime"):
                        start_epoch = int(timestamps["start"].to_pydatetime().timestamp())  # UNIX epoch time
                    else:
                        start_epoch = int(timestamps["start"].timestamp())  # UNIX epoch time

                    os.environ['FONTCONFIG_QUIET'] = '1'

                    filter_complex_str = (
                        f"drawtext=text='%{{pts\\:localtime\\:{start_epoch}\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S}}':"
                        f"x=10:y=H-th-10:font=Arial:fontsize=24:fontcolor=white@1.0:box=1:boxcolor=black@0.5:boxborderw=5"
                    )
          

                    # Generate FFmpeg command
                    ffmpeg_cmd = (
                        ffmpeg
                        .input(video_url, ss=start_time, to=end_time)
                        .output(temp_file, filter_complex=filter_complex_str)
                        .global_args('-loglevel', 'quiet')
                        .global_args('-hide_banner')
                        .global_args('-nostats')
                    )

                    # Run the FFmpeg command
                    ffmpeg_cmd.run(overwrite_output=True)
                else:
                    (
                        ffmpeg
                        .input(video_url, ss=start_time, to=end_time)
                        .output(temp_file, c="copy")
                        .global_args('-loglevel', 'quiet')
                        .run(overwrite_output=True)
                    )
        
        total_offset += duration  # Add current video duration to offset
    # except Exception as e:
    #     print(f"Error processing video {video_url}: {e}")
        


def main(params):
    input_Vids_dict = list_and_sort_files(params["VideoFolder"])
    keys = list(input_Vids_dict.keys())
    if not keys:
        print("No GoPro videos found in the current directory.")
        return

    for key in keys:
        streams = []
        streams.append(gpmf.io.extract_gpmf_stream(input_Vids_dict[key][0]))
        streams.append(gpmf.io.extract_gpmf_stream(input_Vids_dict[key][-1]))

        stream = b''.join(streams)
        utcTimes = list(map(lambda x: datetime.fromisoformat(x.value).timestamp(),gpmf.parse.filter_klv(stream,"GPSU")))
        index_of_first_accurate_timestamp=0
        for i in range(len(utcTimes)-1):
            if utcTimes[i+1]-utcTimes[1]>100000:
                index_of_first_accurate_timestamp=i+1
                break

        #utcTimes = utcTimes[index_of_first_accurate_timestamp:]
        firstTimeStamp=utcTimes[index_of_first_accurate_timestamp]-index_of_first_accurate_timestamp
        lastTimeStamp=utcTimes[-1]-index_of_first_accurate_timestamp
        print("First Datetime present in video: ", datetime.fromtimestamp(firstTimeStamp))
        print("Last Datetime present in video: ", datetime.fromtimestamp(lastTimeStamp))

        utcTimes = utcTimes[index_of_first_accurate_timestamp:]

        # plt.plot(utcTimes)
        # plt.show()

        df = pd.read_csv(params["TurnPeaksFileName"],index_col=0)


        df = df.rename(columns={'start_time': 'start_str', 'end_time': 'end_str'})

        def formatTime(datetimeInput) -> str | int:
            # print(time_str)
            # print(firstDateTime)
            # print(datetimeInput)

            # Convert pandas Timestamp to datetime if needed
            if hasattr(datetimeInput, 'to_pydatetime'):
                datetimeInput = datetimeInput.to_pydatetime()

            timeDiff = datetimeInput.timestamp()-firstTimeStamp

            if timeDiff<0:
                return -1
            seconds = int(timeDiff % 60)    
            minutes = int((timeDiff // 60) % 60)
            hours = int(timeDiff // 3600)
            return f"{hours:02}:{minutes:02}:{seconds:02}"
    

        try:
            df["start"] = pd.to_datetime(df["start_str"])
            df["end"] = pd.to_datetime(df["end_str"])

            df['start_str'] = df['start'].apply(lambda x: np.nan if formatTime(x) == -1 else formatTime(x))
            df['end_str'] = df['end'].apply(lambda x: np.nan if formatTime(x) == -1 else formatTime(x))
            df = df.dropna()
        except Exception as e:
            print(f"Error creating videoTimestamp for {key}: {e}")
            continue

        if params["Turn Numbers"].lower() == "all":
            if params['Type'].lower() != "all":
                df = df[params['Type'].lower()==df['turn_type'].apply(lambda x: x.lower())]
        else:
            params['Turn Numbers'] = np.uint32(params['Turn Numbers'].split(","))
            df = df.loc[params['Turn Numbers']]
        df["turnIndex"] = df.index

        timestamp_pairs = df.to_dict(orient='records')

        create_supercut(input_Vids_dict[key],timestamp_pairs, args={"add_timestamp": params["add_timestamps"].lower()=="true", "add_10s": params["add_10s"].lower()=="true"})





if __name__== "__main__":
    params_txt = "paramsForTurnVideos.txt"

    with open(params_txt, 'r', encoding="utf-16") as file:
        paramList = file.read().split('\n')
        params = {}
        for param in paramList:
            key_param, value = param.split(':')
            params[key_param.strip()] = value.strip()
        
    main(params)