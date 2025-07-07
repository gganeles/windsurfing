# Turn Parser

This tool detects windsurfing turns from GPS data using Sailmon sensor information.

## Features
- Automatically detects tack and jibe transitions from Sailmon data
- Filters turns based on minimum speed requirements
- Classifies turns as Tacks or Jibes
- Finds precise start and end points for each turn
- Outputs start and end time of each turn

## Installation

1. **Install Python 3.7+**
[Download latest version of Python here](https://www.python.org/downloads/)
2. **Open directory in Terminal:**
   Navigate to the folder that contains the file `turnParser.py` using ```cd [directory-name]```
3. **Install dependencies:**

   Run the following command in Terminal once you are in the folder containing the file
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Get your data ready:**
   - Export your session from Sailmon as a CSV file and save it somewhere on your computer.
   - You **MUST** have a file called `paramsForTurnParser.txt` in the folder containing `turnParser.py` with the following content:
     ```
     SailmonCsvFileName: path/to/your_sailmon_file.csv
     ```
        If you don't have this file, create it.
        (Replace `path/to/your_sailmon_file.csv` with the actual path to your Sailmon CSV file, relative to the current folder.)

2. **Run the turn parser:**
   - Open a Terminal window and navigate to the folder containing `turnParser.py`, using the command `cd <folder-name>`.
   - Run this command:
     ```bash
     python turnParser.py
     ```

## Output
- `turns.csv`: Start, end, and peak time of each turn present in the Sailmon data. 
    - Note: this file is needed by `turnAnalysis.py` and `turnVideoGenerator.py`

## Parameters

Default detection parameters (can be changed by editing code):
- **Velocity Threshold**: 8 knots (minimum speed for valid turns)
- **Min Angle**: 33° (minimum acceptable turn angle)Turn Numbers: all
Type: jibe
SailmonCsvFileName: 010324\2024-03-01-sailmon-1 TomR.csv
TurnPeaksFileName: turns.csv
OutputFileName: turn_analysis_results.csv
sort_by: effectiveness
- **Max Angle**: 110° (maximum acceptable turn angle)
- **Window Size**: 10 points (before/after turn for analysis)
- **Error Threshold**: 1e-7 (maximum fitting error for straight segments)

## Notes
- Only Sailmon CSV files with GPS are supported

## Troubleshooting
- Ensure all dependencies are installed
- Check that CSV files contain required columns (GPS, COG, SOG, TWD)

---
For questions or issues, please reach out to Bar Cohen on Whatsapp at +972 52-676-2446