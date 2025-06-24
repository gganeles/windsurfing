import gpmf
import pytz
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import subprocess
import scipy as sp
import matplotlib.pyplot as plt


# if you want to extract the number of timesteps from the video files to speed up, you can use the following command
# ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 input.mp4
def extract_timesteps():
    out = []
    timestep = 1001 / 30000
    for i in range(len(sys.argv)-1-sys.argv.count("-w")):
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',"frame=pts_time", '-of', 'default=nw=1:nk=1', sys.argv[i+1]]
        p = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return 0
        else:
            if i == 0:
                out.extend(list(map(float,p.stdout.splitlines())))
            else:
                out.extend(list(map(lambda x:float(x)+out[-1]+i*timestep,p.stdout.splitlines())))
    return np.array(out)
    
    
    
    

    #(ffmpeg.input(fname)\
    #    .output("pipe:", format="rawvideo", map="0:%i" % stream_index, codec="copy")\
    #    .run(capture_stdout=True, capture_stderr=not verbose)[0])

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
    
    if "SCAL" in block_dict:
        data = block_dict[fourcc].value * 1.0 / block_dict["SCAL"].value
    else:
        data = block_dict[fourcc].value
    
    t = block_dict["STMP"].value
    
    return {"data":data,"t":t}

def quat_multiply(q1, q2):
    """Quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return np.array([w, x, y, z])

def normalize_quaternion(q):
    """Normalize quaternion to unit length."""
    return q / np.linalg.norm(q)

def quaternionGenerate(gyro_data,dt_vec):
    qs = np.empty((gyro_data.shape[0]+1,4))
    qs[0,:]=np.array([1,0,0,0])
    for i in range(gyro_data.shape[0]):
        qs[i+1,:] = normalize_quaternion(qs[i,:]+dt_vec[i]*.5*quat_multiply(qs[i,:],np.concatenate([[0], gyro_data[i]])))
    return qs[1:,:]

def parse_gps_block(gps_block):
    """Turn GPS data blocks into `GPSData` objects

    Parameters
    ----------
    gps_block: list of KVLItem
        A list of KVLItem corresponding to a GPS data block.

    Returns
    -------
    gps_data: GPSData
        A GPSData object holding the GPS information of a block.
    """
    block_dict = {
        s.key: s for s in gps_block
    }

    gps_data = block_dict["GPS5"].value * 1.0 / block_dict["SCAL"].value

    latitude, longitude, altitude, speed_2d, speed_3d = gps_data.T

    return {
        't':block_dict["STMP"].value,
        'description':block_dict["STNM"].value,
        'timestamp':block_dict["GPSU"].value,
        'precision':block_dict["GPSP"].value / 100.,
        'fix':block_dict["GPSF"].value,
        "latitude":latitude,
        'longitude':longitude,
        'altitude':altitude,
        'speed_2d':speed_2d,
        'speed_3d':speed_3d,
        'units':block_dict["UNIT"].value,
        'npoints':len(gps_data)
    }



if "-w" in sys.argv:
    tvec = extract_timesteps()
    print(tvec)
    if tvec == 0:
        print("Error extracting timesteps")
        exit(1)
    rawDF = pd.DataFrame({
        "t":tvec
    })

    rawDF.to_csv("rawData.csv")
else:
    rawDF = pd.read_csv("rawData.csv")
    tvec = rawDF["t"].to_numpy()

streams = []
if (len(sys.argv)>1):
    for x in sys.argv[1:]:
        if x=="-w":
            continue
        streams.append(gpmf.io.extract_gpmf_stream(x))
else:
    print("Please enter lrv file names in order")
    exit(1)

stream = b''.join(streams)

print(gpmf.parse.expand_klv(stream)[0])

def extractData(stream, fourcc):
    return pd.DataFrame(list(map(lambda x: pythonify_block(x,fourcc),extract_blocks(stream,fourcc))))

accl = extractData(stream, "ACCL")       

gyro = extractData(stream, "GYRO")

gps5 = pd.DataFrame(list(map(parse_gps_block, extract_blocks(stream, "GPS5"))))

grav = extractData(stream, "GRAV")

lowest_t = min([gps5["t"].iloc[0],accl["t"].iloc[0],gyro["t"].iloc[0],grav["t"].iloc[0]])

highest_t = tvec[-1]*1000000

timestep = 1001000/30000

t = np.append(np.arange(0,(highest_t)/1000,timestep),highest_t/1000)

def interpTime(df):
    
    def generate_sequence(start, interval, amount):
        """
        Generate a sequence of numbers starting at `start`, with a step of `interval`, and `amount` numbers.

        Parameters:
            start (float): The starting number of the sequence.
            interval (float): The step size between consecutive numbers.
            amount (int): The total number of numbers in the sequence.

        Returns:
            list: A list of generated numbers.
        """
        return list(np.arange(start, start + interval * amount, interval)[:amount])
    
    times = []
    numSamps = []
    for i in range(df.shape[0]-1):
        numSamps.append(df["data"].iloc[i].shape[0])
        times.append(np.linspace(df["t"].iloc[i],df["t"].iloc[i+1],df["data"].iloc[i].shape[0],endpoint=False))
    final_pkt_samps = df["data"].iloc[-1].shape[0]
    time_per_samp = np.mean(np.diff(df["t"]))/np.mean(numSamps)
    times.append(generate_sequence(df["t"].iloc[-1],time_per_samp,final_pkt_samps))
    return np.concatenate(times)

def streamTo29(stream,fourcc,t,lowest_t):
    df = extractData(stream,fourcc)
    timev = interpTime(df)
    print(df['data'])
    data = np.concatenate(df["data"])
    
    output = np.empty((t.shape[0],data.shape[1]))
    for i in range(data.shape[1]):
        output[:,i] = np.interp(t, (timev-lowest_t)/1000,data[:,i])
    return output

accl_t = interpTime(accl)
accl_concat = np.concatenate(accl["data"])
    
grav_t = interpTime(grav)
grav_concat = np.concatenate(grav["data"])

accl29 = np.empty((t.shape[0],3))
grav29 = np.empty((t.shape[0],3))

for i in range(3):
    #accl_spline = sp.interpolate.CubicSpline((accl_t-lowest_t),accl_concat[:,i])
    #accl29[:,i] = accl_spline(t*1000)
    accl29[:,i] = np.interp(t, (accl_t-lowest_t)/1000,accl_concat[:,i])

    grav29[:,i] = np.interp(t, (grav_t-lowest_t)/1000,grav_concat[:,i]) #matches data well
    

gyro_t = interpTime(gyro)
gyro_concat = np.concatenate(gyro["data"])

#quats=quaternionGenerate(gyro_concat, usecPerSamp/(numSamps*1000000))
gyro_dt = np.diff(gyro_t)

quats = quaternionGenerate(gyro_concat, np.append(gyro_dt,gyro_dt[-2])/1000000)

quats29 = np.empty((t.shape[0],4))
for i in range(4):
    quats29[:,i] = np.interp(t, (gyro_t-lowest_t)/1000,quats[:,i])

gpsTimeData = pd.DataFrame({"data":gps5["latitude"],"t":gps5["t"]})
gpst10  = interpTime(gpsTimeData)
#gpst10 = np.linspace(gps5["t"].iloc[0],gps5["t"].iloc[-1],np.concatenate(gps5["longitude"]).shape[0])

gpst = (gps5["t"]-lowest_t)/1000 

timeU = np.interp(t,gpst,list(map(lambda x: datetime.fromisoformat(x).timestamp(),gps5["timestamp"])))
unixTime = list(map(lambda x: pytz.UTC.localize(datetime.fromtimestamp(x)).astimezone(pytz.timezone("Asia/Jerusalem")).replace(tzinfo=None),timeU))

def gpsInterp(t,gpst10,gps5_field):
    return np.interp(t,(gpst10-lowest_t)/1000+5,np.concatenate(gps5_field))

mwet = streamTo29(stream,"MWET",t,lowest_t)

quats2 = streamTo29(stream,"CORI",t,lowest_t)

df = pd.DataFrame({
    "video-time (ms)":t,
    "unix-time [utc] (date)":unixTime,
    "latitude (deg)": gpsInterp(t,gpst10,gps5["latitude"]),
    "longitude (deg)":gpsInterp(t,gpst10,gps5["longitude"]),
    "altitude [gps msl] (m)":gpsInterp(t,gpst10,gps5["altitude"]),
    "speed [2d gps] (m/s)":gpsInterp(t,gpst10,gps5["speed_2d"]),
    "orientation-quaternion-w":quats2[:,0],
    "orientation-quaternion-x":quats2[:,3],
    "orientation-quaternion-y":quats2[:,1],
    "orientation-quaternion-z":quats2[:,2],
    "accelerometer-x (m/s²)":accl29[:,1],
    "accelerometer-y (m/s²)":accl29[:,2],
    "accelerometer-z (m/s²)":accl29[:,0],
    "gravity-vector-x":grav29[:,1],
    "gravity-vector-y":grav29[:,2],
    "gravity-vector-z":grav29[:,0]
})

df.to_csv("GL_New_"+sys.argv[2][4:8]+".csv",index=False)

print("Saving data to GL_New_"+sys.argv[2][4:8]+".csv")

exit(0)