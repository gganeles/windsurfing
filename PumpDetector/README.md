# GoPro Pump Detector

This tool detects pumping events in windsurfing GoPro videos using embedded sensor data.

## Features
- Extracts and analyzes acceleration and GPS data from GoPro videos
- Detects pumping events and outputs their timestamps
- Creates a short video of each detected pumping event

## Installation

1. **Install Python 3.7+**
   [Download latest version of Python here](https://www.python.org/downloads/)
2. **Install FFMPEG:**
   [Download FFMPEG here](https://ffmpeg.org/download.html)
3. **Open directory in Terminal:**
   Navigate to the folder that contains the file `goProPumpingParser.py` using ```cd [directory-name]```
4. **Install dependencies:**
   Run the following command in Terminal once you are in the folder containing the file
   ```bash
   pip install -r requirements.txt
   ```



## Usage

1. **Prepare your files:**
   - Place your GoPro video files in the same folder as this script. Make sure they all have their original names, e.g. GL010010.LRV.
   - A `params.txt` file **MUST** be present in the same folder as this script (see below for format).

2. **Configure parameters:**
   - If desired, edit `params.txt` to adjust detection thresholds and options. Instructions listed below.

3. **Run the tool:**
   Run the following command in Terminal once you are in the folder containing the file
   ```bash
   python goProPumpingParser.py
   ```

## Output
- CSV (excel) files with detected pump beginning and end timestamps are saved in folders named after each video.
- Video clips of detected events are saved in the same folders.

## Parameters (`params.txt`)

 - In general, to increase the number of pumps detected, lower the ```pumpThresh``` by editing the value in the `params.txt` file, then saving it

Default `params.txt` file:
```
accelCutOff: 35
lowPassCutoff: 0.4
fs: 201
kernel_n (sec): 15
pumpThresh: 1.7
fine_kernel_n (sec): 5
finePumpThresh: 0.15
velocityThresh: 4
plots: False
add10sec: False
addTimestamps: True
videoQuality: medium
```

### Parameter Meanings
Edit any of the parameters in the `params.txt` file as desired.
- **accelCutOff**: Acceleration (in m/s^2) above this threshold is ignored (makes processing easier)
- **lowPassCutoff**: Low-pass filter cutoff frequency (normalized frequency)
- **fs**: Gopro Accelorometer Sampling rate (Hz)
- **kernel_n (sec)**: Window size for main moving average (seconds)
- **pumpThresh**: Absolute acceleration threshold for pump detection. Anything above this acceleration value is considered a pump.
- **fine_kernel_n (sec)**: Window size for smaller moving average (seconds)
- **finePumpThresh**: Threshold for pump edge refinement
- **velocityThresh**: Minimum average velocity to be considered a valid pump
- **plots**: Set to `True` to show debug plots
- **add10sec**: Set to `True` to extend clips by 10 seconds before/after
- **addTimestamps**: Set to `True` to overlay timestamps on output videos
- **videoQuality**: Video encoding quality level (see Video Quality Options below)

### Video Quality Options
The `videoQuality` parameter controls the balance between video quality, file size, and processing time:

- **"low"**: Fast processing, smaller files (good for previews)
- **"medium"**: Balanced quality and speed (recommended for most uses) [DEFAULT]
- **"high"**: Better quality, slower processing (good for final videos)
- **"very high"**: Best quality, slowest processing (archive quality)
- **"copy"**: No re-encoding, fastest but limited editing capability

## Notes
- Only GoPro videos with embedded sensor data (GPMF) are supported.

## Troubleshooting
- Ensure all dependencies are installed.
- Ensure FFmpeg is installed and accessible in your system PATH
- Make sure `params.txt` is present and correctly formatted.
- Place video files in the same directory as the script with their original names.

---
For questions or issues, please text Gabriel Ganeles on Whatsapp at [+972 58-712-0601](https://wa.me/qr/EWMOYZYAUGN6D1)
