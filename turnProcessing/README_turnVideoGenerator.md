# Turn Video Generator

This tool automatically extracts video clips from GoPro footage based on detected windsurfing turns. It uses GPS metadata from GoPro videos to align the timestamps and create videos of specific turns or all turns from a session.

## Features
- Extracts video clips from GoPro videos based on turn timestamps
- Handles sessions split across multiple GoPro files
- Can process all turns or specific turn numbers
- Optionally adds timestamp overlays to video clips
- Can add buffer time before/after turns

## Installation

1. **Install Python 3.7+**
   [Download latest version of Python here](https://www.python.org/downloads/)

2. **Install FFMPEG:**
   [Download FFMPEG here](https://ffmpeg.org/download.html)

3. **Open directory in Terminal:**
   Navigate to the folder that contains the file `turnVideoGenerator.py` using ```cd [directory-name]```

4. **Install dependencies:**
   Run the following command in Terminal once you are in the folder containing the file
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your files:**
   - A `paramsForTurnVideos.txt` file **MUST** be present in the same folder as this script (see below for format)
   - Place your GoPro video files in a folder (e.g., "VideoFolder").
   - This tool assumes GoPro video files follow the standard naming convention (e.g., "GL010095.LRV")
   - Make sure you have a `turns.csv` file with turn timestamps (created by `turnParser.py`)

2. **Configure parameters:**
   - Edit `paramsForTurnVideos.txt` to specify your video folder and turn options

3. **Run the tool:**
   Run the following command in Terminal once you are in the folder containing the file
   ```bash
   python turnVideoGenerator.py
   ```

## Output
- Creates numbered MP4 video clips for each turn
- Organizes clips in folders named by GoPro session ID (e.g. if input is GL010095.LRV, output will be a folder called GL0095)
- Video filename format: `{gopro_session_id}_{turn_number}.mp4` e.g. `GL0095_1.mp4`

## Parameters (`paramsForTurnVideos.txt`)

Default `paramsForTurnVideos.txt` file:
```
Turn Numbers: all
Type: all
TurnPeaksFileName: path/to/your/turns.csv
VideoFolder: path/to/your/video/folder
add_10s: True
add_timestamps: True
```

### Parameter Meanings
Edit any of the parameters in the `paramsForTurnVideos.txt` file as desired.
- **VideoFolder**: Directory containing your GoPro video files
- **TurnPeaksFileName**: path to your turns CSV file (created by `turnParser.py`)
- **Turn Numbers**: Either "all" or comma-separated list of turn numbers (e.g., "0,3,5")
- **Type**: Filter by turn type ("Tack" or "Jibe"). Only works if Turn Numbers = 'all'.
- **add_10s**: Add 10 seconds before and after each turn (True/False)
- **add_timestamps**: Add timestamp overlay to videos (True/False)


## Input File Requirements

### Turn Peaks File
Your turns CSV file must contain these columns:
- `start_time`: Start datetime of the turn
- `end_time`: End datetime of the turn
- `turn_type`: Type of turn ("Tack" or "Jibe")

### GoPro Video Files
- Must follow standard GoPro naming convention (GL010095.MP4, GL010095.LRV, etc.)
- Must contain GPS metadata (GPMF) for accurate timing
- Can be split across multiple files for long sessions

## Notes
- Only GoPro videos with embedded GPS metadata are supported
- The Sailmon data usually exceeds the length of the GoPro data, therefore not all turns are videoed
- Ensure sufficient disk space for output clips
- Ignore outputs about font files

## Troubleshooting
- Ensure FFmpeg is installed and accessible in your system PATH
- Check that your video folder path is correct
- Be sure to use original GoPro files

---
For questions or issues, please text Gabriel Ganeles on Whatsapp at [+972 58-712-0601](https://wa.me/qr/EWMOYZYAUGN6D1) 