# Turn Analysis Tool

This tool analyzes windsurfing turn effectiveness by calculating performance metrics from GPS data. It uses Sailmon sensor data and turn timestamps (from `turnParser.py`) to evaluate how well each turn was executed.

## Features
- Calculates turn effectiveness and lost distance during turns
- Analyzes the effectiveness of the exit after completing turns
- Optional GPS data visualization for detailed turn analysis

## Installation

1. **Install Python 3.7+**
   [Download latest version of Python here](https://www.python.org/downloads/)

2. **Open directory in Terminal:**
   Navigate to the folder that contains the file `turnAnalysis.py` using ```cd [directory-name]```

3. **Install dependencies:**

   Run the following command in Terminal once you are in the folder containing the file
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare your files:**
   - A `paramsForTurnAnalysis.txt` file **MUST** be present in the same folder as this script (see below for format)
   - Make sure you have a CSV file containing data exported from a Sailmon
   - Make sure you have a `turns.csv` file with turn timestamps. This is created by using the script `turnParser.py`

2. **Configure parameters:**
   - Edit `paramsForTurnAnalysis.txt` to specify where to find your input files, as well as which turns to analyze

3. **Run the tool:**

   Run the following command in Terminal once you are in the folder containing the file
   ```bash
   python turnAnalysis.py
   ```

## Output
- CSV file with turn analysis results. For each turn, the tool calculates:
  - **Turn Length**: How long the turn took
  - **Lost Distance**: Distance lost during the turn
  - **Effectiveness**: How well the turn was executed (higher is better)
  - **Exit Effectiveness**: How well you performed after completing the turn

    
- Results are sorted by effectiveness in descending order
- One can use the file `plotTurn.py` in order to look at the details of specific turns by running:
  ```bash
  python plotTurn.py <turn numbers or 'all'>
  ```
  This will generate visualizations showing the GPS path, velocity made good, and true wind angles for each turn, helping you understand what happened during specific turns and why certain turns performed better or worse than others.

## Parameters (`paramsForTurnAnalysis.txt`)

Default `paramsForTurnAnalysis.txt` file:
```
Turn Numbers: all
Type: all
SailmonCsvFileName: path/to/your/sailmon_data.csv
TurnPeaksFileName: path/to/your/turns.csv
OutputFileName: turn_analysis_results.csv
sort_by: effectiveness
```

### Parameter Meanings
Edit any of the parameters in the `paramsForTurnAnalysis.txt` file as desired.
- **Turn Numbers**: Either "all" or comma-separated list of turn numbers (e.g., "0,3,5")
- **Type**: Filter by turn type ("Tack" or "Jibe"). Only works if Turn Numbers = 'all'.
- **SailmonCsvFileName**: Name of your Sailmon CSV file
- **TurnPeaksFileName**: Name of your turns CSV file (created by running `turnParser.py`)
- **OutputFileName**: Desired name for the output results file. 
- **sort_by**: Which column to sort the results by in the output CSV. Default is "effectiveness". You can also use "turn_index", "lost_distance", etc.



## Input File Requirements

### Sailmon CSV File
Your Sailmon CSV file must contain these columns:
- `time`: Timestamp of each measurement
- `latitude`, `longitude`: GPS coordinates
- `COG - Course over Ground`: Course over ground in degrees
- `SOG - Speed over Ground`: Speed over ground in m/s
- `TWD - True Wind Direction`: True wind direction in degrees

### Turn Peaks File
Your turns CSV file must contain these columns:
- `peak_time`: Timestamp of turn peak
- `turn_type`: Type of turn ("Tack" or "Jibe")

## Notes
- The tool assumes a constant True Wind Direction of 90 degrees
- Very buggy. Negative effectiveness and lost distance values are common, despite having unclear meaning. Perhaps plot turns in these cases to assess effectiveness.
- One can use the file `plotTurn.py` in order to plot the details of specific turns by running:
  ```bash
  python plotTurn.py <turn numbers or 'all'>
  ```
  This will generate visualizations showing the GPS path, velocity made good, and true wind angles for each turn, helping you understand what happened during specific turns and why certain turns performed better or worse than others.


## Troubleshooting
- Ensure all required columns are present in your input csv, and that it is correctly formatted
- Check that timestamps in your turns file match the Sailmon time range
- Make sure all dependencies are installed
- Use the `plotTurn.py` script to inspect specific problematic turns

---
For questions or issues, please text Gabriel Ganeles on Whatsapp at [+972 58-712-0601](https://wa.me/qr/EWMOYZYAUGN6D1)