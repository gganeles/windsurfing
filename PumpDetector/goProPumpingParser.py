import gpmf
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ffmpeg
from scipy.signal import convolve, butter, filtfilt

from datetime import datetime, timedelta
from collections import defaultdict
import re
import os
import sys
import copy
import os



 


def get_resource_path(relative_path):
    """ Get the absolute path to a resource, compatible with PyInstaller. """
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = sys._MEIPASS
    else:
        # Running as script
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def window_rms(a, window_size):
    a2 = np.power(a,2)
    window = np.ones(window_size)/float(window_size)
    return np.sqrt(np.convolve(a2, window, 'same'))


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
    
    
    
def main(params):
    videoNames = list_and_sort_files(get_resource_path("./"))
    add_timestamp = params['addTimestamps'].lower() == "true"
    add_10s = params['add10sec'].lower() == "true"
    for x in videoNames:
        output = mainPart(videoNames[x],"pumps"+x+".csv",params)
        if __name__!="__main__":
            return output
        else:
            if len(output)>0:
                create_supercut(videoNames[x],output,args={"add_timestamp":add_timestamp,"add_10s":add_10s})

def mainPart(filenames,filename,params):
    # Define Acceleration filter cutoff
    accel_raw_cutoff = float(params["accelCutOff"])
    
    # Define passband
    lowPassCutoff = float(params["lowPassCutoff"])
    
    # Define sampling rate
    fs = int(params["fs"])
    
    # Kernel definitions
    kernel_n = int(params["kernel_n (sec)"])*fs
    
    # Pumping threshold
    pumpThresh = float(params["pumpThresh"])

    kernel_n_2 = int(params["fine_kernel_n (sec)"])*fs

    plots = params["plots"] == "true"
    
    finePumpThresh = float(params["finePumpThresh"])

    velocityThresh = float(params["velocityThresh"])

    print(f"\nProcessing video {filenames[0][4:8]}")
    
    streams = []
    for x in filenames:
        streams.append(gpmf.io.extract_gpmf_stream(x))

    stream = b''.join(streams)
    #print(gpmf.parse.expand_klv(stream))
    
    
    #Extract GPS low level data from the stream
    gps_blocks = gpmf.gps.extract_gps_blocks(stream)


    accel_blocks = extract_blocks(stream,"ACCL")

    devName = list(gpmf.parse.filter_klv(stream,"DVNM"))[0].value

    devID = list(gpmf.parse.filter_klv(stream,"DVID"))[0].value


    print("Camera Name: ",devName, devID)

    # Parse low level data into more usable format
    gps_data = list(map(gpmf.gps.parse_gps_block, gps_blocks))

    accel_data = list(map(lambda x: pythonify_block(x,"ACCL")['data'],accel_blocks))
  
    utcTimes = list(map(lambda x: datetime.fromisoformat(x.value).timestamp(),gpmf.parse.filter_klv(stream,"GPSU")))

    index_of_first_accurate_timestamp=0
    for i in range(len(utcTimes)-1):
        if utcTimes[i+1]-utcTimes[1]>1000000:
            index_of_first_accurate_timestamp=i+1
            break

    utcTimes = utcTimes[index_of_first_accurate_timestamp:]

    velocity = list(map(lambda x: x.speed_2d,gps_data))

    velocity_concat = np.concatenate(velocity[index_of_first_accurate_timestamp:])
 
    accel_data_concat = np.concatenate(accel_data[index_of_first_accurate_timestamp:])

    accel_n = accel_data_concat.shape[0]

    abs_accel = np.zeros(accel_n)

    for i in range(accel_n):
        abs_accel[i]=((accel_data_concat[i,0]**2+accel_data_concat[i,1]**2+accel_data_concat[i,2]**2)**.5)

    accel_timestamps = np.interp(np.linspace(0,len(utcTimes),accel_n),range(len(utcTimes)),utcTimes)
    velocity_t =np.linspace(0,len(velocity_concat),accel_n)
    velocity_interp = np.interp(velocity_t,range(len(velocity_concat)),velocity_concat)

    timevec = np.array(list(map(lambda x: pytz.UTC.localize(datetime.fromtimestamp(x)).astimezone(pytz.timezone("Asia/Jerusalem")).replace(tzinfo=None),accel_timestamps)))

    # Apply a cutoff filter

    abs_accel_cut = np.zeros(abs_accel.shape[0])
    for i in range(abs_accel.shape[0]):
        abs_accel_cut[i] = np.where(abs_accel[i] < accel_raw_cutoff, abs_accel[i], 0)

    # Lowpass filter
    b, a = butter(4, lowPassCutoff, btype='low', fs=fs)
    lowpassed_accel = filtfilt(b, a, abs_accel_cut)
    lowpassed_accel = lowpassed_accel - np.mean(lowpassed_accel)


    # light lowpass filter
    # b, a = butter(4, .8, btype='low', fs=fs)
    # weak_lowpassed_accel = filtfilt(b, a, abs_accel_cut)   
    # weak_lowpassed_accel = weak_lowpassed_accel - np.mean(weak_lowpassed_accel)
    # Bandpass filter
    # b_band, a_band = butter(2, passBand, btype='band', fs=fs)
    # bandpassed_accel = filtfilt(b_band, a_band, abs_accel_cut)

    # shift lowpassed_accel down
 
    
    # plt.plot(timevec,lowpassed_accel)
    # plt.show()

    # Apply Hilbert transform
    # hilb_accel_z = hilbert(lowpassed_accel)

    # Kernel definitions
    
    h = np.ones(kernel_n) / kernel_n

    x = np.linspace(-np.pi, np.pi, kernel_n)
    # h1 = (np.cos(kernel_n*x) + 1) / (2*kernel_n)
    # h2 = -np.sinc(x / np.pi) / kernel_n

    h3 = (1 / kernel_n) * (np.concatenate([
        np.ones(kernel_n // 3),
        0.1 * np.ones(kernel_n // 3),
        np.ones(kernel_n // 3 + kernel_n % 3)
    ]))
        
    def plot_normed_convolution(x,kernel,timevec,label):
        output = convolve(x,kernel,mode='same')
        output = output - np.mean(output)
        #output = output * (1/max(output))
        n=kernel.shape[0]
        #plt.plot(timevec,output[:output.shape[0]-n+1],label=label)
        #return output[:output.shape[0]-n+1]
        if plots:
            plt.plot(timevec,output,label=label)
        return output
    
    # Convolutions
    if plots:
        plt.figure()
    
    
    UnHilb_avg_conv = plot_normed_convolution(np.abs(lowpassed_accel), h,timevec, r"First Pass")
    
    #plot_normed_convolution(np.abs(weak_lowpassed_accel),np.array([1]),timevec,"Finer Detail")
    
    #plot_normed_convolution(lowpassed_accel,np.array([1]),timevec,"Fine Detail")

    #cos_conv_accel = plot_normed_convolution(np.abs(hilb_accel_z), h1,timevec, f"cos abs hilb")
    #avg_conv_accel = plot_normed_convolution(np.abs(hilb_accel_z), h,timevec, f"avg abs hilb")
    #h3_conv_accel = plot_normed_convolution(np.abs(hilb_accel_z), h3,timevec, f"h3 abs hilb")
    #h3_abs_conv_accel = plot_normed_convolution(np.abs(lowpassed_accel),h3,timevec, "h3 abs Lowpass")
    #rms = window_rms(np.abs(lowpassed_accel),kernel_n)
    #rms = 1/np.max(rms) * rms
    #plt.plot(timevec,rms,label="RMS")
    

    

    if plots:
        plt.title(f"Wide Convolution of Acceleration Data for {filenames[0][4:8]}")
        plt.show()

    if plots:
        #plt.plot(timevec, abs_accel_cut)
        #plt.plot(timevec,gps_data_interp,label="gps velocity")
        plot_normed_convolution(np.abs(lowpassed_accel), h,timevec, r"Acceleration")
        plt.plot([timevec[0],timevec[-1]],[pumpThresh,pumpThresh],label="Pump Threshold",linestyle='--')
        
    pumps = extractPumps(UnHilb_avg_conv,pumpThresh)
    averagedLowpass5 = plot_normed_convolution(np.abs(lowpassed_accel), 1/kernel_n_2*np.ones(kernel_n_2), timevec, "Lowpass 5s Kernel")

    #refinePumps(lowpassed_accel,pumps,velocity_concat,4)


    oldPumps = copy.deepcopy(pumps)
    pumps = refinePumps2(averagedLowpass5,pumps,finePumpThresh,velocity_interp,velocityThresh)

    addPumpMetaData(pumps,timevec,index_of_first_accurate_timestamp)
    if plots and len(pumps):
        #plt.plot(timevec,lowpassed_accel,label="Lowpass Acceleration")
        plt.plot(timevec,velocity_interp, label="GPS Velocity")
        plotPumpBounds(pumps,timevec,averagedLowpass5)
        plotPumpBounds(oldPumps,timevec,UnHilb_avg_conv)
        plt.legend()
        from labeledpumps010 import pumps as LabeledPumps
        for x in LabeledPumps:
            x_1=datetime.fromisoformat(x["start"])
            x_2=datetime.fromisoformat(x["end"])
            plt.axvline(x=x_1, color='r', linestyle='--')
            plt.axvline(x=x_2, color='r', linestyle='--')
        plt.show()

    if len(pumps)==0:
        print("No pumps found in video "+filenames[0][4:8])
        return []
    
    output = pd.DataFrame(pumps)
    # Get video duration
    # Create a directory for the video name if it doesn't exist
    video_name = os.path.splitext(os.path.basename(filenames[0]))[0]
    video_dir = os.path.join(os.getcwd(), video_name[0:2]+video_name[4:])
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)



    output["duration"] = output.apply(lambda row: (row["end"] - row["start"]).total_seconds(), axis=1)
    output["start"] = output["start"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    output["end"] = output["end"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    output["videostartstamp"] = output["start_str"]
    output["videoendstamp"] = output["end_str"]
    outputFile = output[["start", "end", "videostartstamp", "videoendstamp", "duration"]]
    outputFile.to_csv(video_dir+"\\"+filename)

    return pumps

def refinePumps2(data,pumps,thresh,vel,velThresh=4):
    refined = []
    for pump in pumps:
        startInd = pump["startInd"]
        endInd = pump["endInd"]
        while data[startInd]<thresh and startInd!=endInd:
            startInd+=1
        if startInd==endInd:
            continue
        while data[endInd]<thresh:
            endInd-=1
        if np.mean(vel[startInd:endInd])>velThresh:
            pump["startInd"]=startInd
            pump["endInd"]=endInd
            refined.append(pump)
    return refined

def extract_blocks(stream,fourcc):
    """ Extract GPS data blocks from binary stream

    This is a generator on lists `KVLItem` objects. In
    the GPMF stream, GPS data comes into blocks of several

    different data items. For each of these blocks we return a list.

    Parameters
    ----------
    stream: bytes
        The raw GPMF binary stream

    Returns
    -------
    gps_items_generator: generator
        Generator of lists of `KVLItem` objects
    """
    for s in gpmf.parse.filter_klv(stream, "STRM"):
        content = []
        is_gps = False
        for elt in s.value:
            content.append(elt)
            if elt.key == fourcc:
                is_gps = True
        if is_gps:
            yield content

def pythonify_block(block,fourcc):
    block_dict = {
        s.key: s for s in block
    }
    data = block_dict[fourcc].value * 1.0 / block_dict["SCAL"].value
    
    t = block_dict["STMP"].value
    
    return {"data":data,"t":t}

def extractPumps(data,pumpThresh):
    pumps = []
    pumpFlag = False
    n = data.shape[0]
    for i in range(n):
        if pumpFlag:
            if data[i] < pumpThresh and data[i]>data[i+1]:
                pumpFlag = False
                pumps[-1]["endInd"]=i
        else:
            if data[i] > pumpThresh and data[i]<data[i+1]:
                pumpFlag=True
                pumps.append({"startInd":i,"endInd":n})
    deriv = np.diff(data)
    for i in range(len(pumps)):
        x = pumps[i]["startInd"]
        while x<len(deriv) and deriv[x]>0:
            x-=1
        pumps[i]["startInd"]=x

    
        x2 = pumps[i]["endInd"]
        while x2<len(deriv) and deriv[x2]<0:
            x2+=1
        pumps[i]["endInd"]=x2
    return pumps

def addPumpMetaData(pumps,timevec,firstTimeIndex):
    firstTime=datetime.fromtimestamp(timevec[0].timestamp()-firstTimeIndex)
    for i in range(len(pumps)):
        pumps[i]["end_str"]=vidTimeStamp(firstTime,timevec[pumps[i]["endInd"]])
        pumps[i]["end"]=timevec[pumps[i]["endInd"]]
        pumps[i]["start"]=timevec[pumps[i]["startInd"]]
        pumps[i]["start_str"]=vidTimeStamp(firstTime,timevec[pumps[i]["startInd"]])
    return pumps

def plotPumpBounds(pumps,timevec,data):
    for pump in pumps:
        plt.scatter(timevec[pump["startInd"]],data[pump["startInd"]])
        plt.scatter(timevec[pump["endInd"]],data[pump["endInd"]])

def refinePumps(data,pumpList,velocity,velocityThresh=4):
    removeList=[]
    for i in range(len(pumpList)):
        if (np.mean(velocity)<velocityThresh):
            removeList.append(i)
        newStart = int((pumpList[i]["startInd"]+pumpList[i]["endInd"])/2)
        newEnd = newStart

        while data[newStart]>0:
            newStart-=1
        while data[newEnd]>0:
            newEnd+=1
        
        pumpList[i]["startInd"] = newStart
        pumpList[i]["endInd"] = newEnd
        if newEnd-newStart<402:
            removeList.append(i)
    for i in removeList[::-1]:
        pumpList.pop(i)
    return pumpList

def vidTimeStamp(datetime1,datetime2):
    # Calculate the time difference
    time_difference = datetime2 - datetime1

    # Extract total seconds
    total_seconds = int(time_difference.total_seconds())

    # Convert to minutes and seconds
    minutes, seconds = divmod(total_seconds, 60)

    hours, minutes = divmod(minutes, 60)

    return f"{hours:02}:{minutes:02}:{seconds:02}"

def hhmmss_to_seconds(hhmmss):
    t = datetime.strptime(hhmmss, "%H:%M:%S")
    return t.hour * 3600 + t.minute * 60 + t.second

def generate_video_ts_vector(datetime_vector, start_datetime, start_time_str):
    def time_str_to_seconds(time_str):
        hh, mm, ss = map(int, time_str.split(":"))
        return hh * 3600 + mm * 60 + ss

    # Helper to convert seconds to video timestamp format hh:mm:ss,mmm
    def seconds_to_video_ts(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        ms = int((s - int(s)) * 1000)  # milliseconds
        return f"{h:02}:{m:02}:{int(s):02},{ms:03}"
    # Convert the start time string to seconds
    start_time_seconds = time_str_to_seconds(start_time_str)
    
    # Calculate the start datetime in seconds
    start_seconds = time_str_to_seconds(start_datetime.strftime("%H:%M:%S"))
    start_offset = start_seconds - start_time_seconds

    # Generate the video timestamps
    video_ts_vector = []
    for dt in datetime_vector:
        # Calculate the difference in seconds between the start datetime and the current datetime
        diff_seconds = (dt - start_datetime).total_seconds()
        # Add the offset to the start time
        video_ts_vector.append(seconds_to_video_ts(start_offset + diff_seconds))
    
    return video_ts_vector


def create_supercut(video_urls, timestamp_pairs, time_offset=0 , args={}):
    total_offset = 0  # Keeps track of cumulative video duration
    try:
        for i, video_url in enumerate(video_urls):
            # Get video duration
            # Create a directory for the video name if it doesn't exist
            video_name = os.path.splitext(os.path.basename(video_url))[0]
            video_dir = os.path.join(os.getcwd(), video_name[0:2]+video_name[4:])



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
                    temp_file = os.path.join(video_dir, f"{video_name[0:2]+video_name[4:]}_{j+1}.mp4")

                    if args.get("add_timestamp"):
                        # Create timestamp text overlay for each timestamp in the vector
                        # Construct drawtext with pts:localtime 

                        start_epoch = int(timestamps["start"].timestamp())  # UNIX epoch time


                        filter_complex_str = (
                            f"drawtext=text='%{{pts\\:localtime\\:{start_epoch}\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S}}':"
                            f"x=10:y=H-th-10:font=Arial:fontsize=24:fontcolor=white@1.0:box=1:boxcolor=black@0.5:boxborderw=5"
                        )

                        # Generate FFmpeg command
                        ffmpeg_cmd = (
                            ffmpeg
                            .input(video_url, ss=start_time, to=end_time)
                            .output(
                                temp_file,
                                format='mp4',
                                filter_complex=filter_complex_str,
                                vcodec='libx264',     # re-encode video
                                acodec='aac',
                                dcodec='copy',  # copy data streams
                                **{
                                    'map': '[v]',         # filtered video
                                    'map': '0:1?',        # audio
                                    'map': '0:3?',        # gpmd
                                    'c:v': 'libx264',
                                    'c:a': 'aac',
                                    'c:d': 'copy',
                                }     # Copy data streams (0:2 and 0:3)
                            )  
                            .global_args('-loglevel', 'error')
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
    except Exception as e:
        print(f"Error processing video {video_url}: {e}")
        


if __name__=="__main__":
    os.path.exists(get_resource_path("params.txt")) or exit("params.txt not found in the current directory. Please provide a valid params.txt file.")

    with open(get_resource_path("params.txt"), 'r') as f:
        paramList = f.read().strip().split('\n')
        params = {}
        for param in paramList:
            key, value = param.split(':')
            params[key.strip()] = value.strip()
    main(params)