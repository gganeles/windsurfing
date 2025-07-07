#!/usr/bin/env python
# coding: utf-8

# In[13]:

import numpy as np
import pandas as pd
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import mean_squared_error
from datetime import timedelta

# In[14]:


with open('paramsForTurnParser.txt', 'r') as file:
    for line in file:
        if line.startswith('SailmonCsvFileName:'):
            SailmonCsvFileName = line.split(':')[1].strip()
    if not SailmonCsvFileName:
        raise ValueError("SailmonCsvFileName is not set in paramsForTurnParser.txt")
    


Sailmon = pd.read_csv(SailmonCsvFileName)
Sailmon


# In[15]:


Sailmon ['TWD - True Wind Direction'] = Sailmon['TWD - True Wind Direction'].ffill().bfill()

#Sailmon ['TWA - True Wind Angle'] = Sailmon['TWA - True Wind Angle'].fillna(method='ffill').fillna(method='bfill')

Sailmon


# In[16]:


sailmon_cut = Sailmon
#sailmon_cut = Sailmon[0:4117]
sailmon_cut


# In[ ]:


#sailmon_cut.to_csv('sailmon_cut.csv', index=False)


# In[269]:


# Step 1: Convert time columns to UTC
# Ensure the column is in a datetime format
sailmon_cut['servertime'] = pd.to_datetime(sailmon_cut['servertime'], format='mixed')

# Convert the `servertime` column to UTC timezone
sailmon_cut['servertime_utc'] = sailmon_cut['servertime'].dt.tz_localize('Asia/Jerusalem').dt.tz_convert('UTC')
sailmon_cut


# In[270]:


# Conversion factor: 1 m/s = 1.94384 knots
sailmon_cut['SOG [knots]'] = sailmon_cut['SOG - Speed over Ground'] * 1.94384
sailmon_cut['sog_smooth[knots]'] = sailmon_cut['SOG [knots]'].rolling(window=15, center=True).mean()
sailmon_cut


# In[270]:





# In[271]:


sailmon_cut


# In[272]:


def detrmine_DW_UW_R_L(COG, TWD):
    """
    Determine relative wind (Upwind/Downwind), tack (Right/Left), multiplier, and angle difference.

    Args:
        COG (float): Course Over Ground in degrees (0-360)
        TWD (float): True Wind Direction in degrees (0-360)

    Returns:
        tuple: (rel_wind, tack, multiplier, angle_diff)
            rel_wind: 'UW' (upwind) or 'DW' (downwind)
            tack: 'R' (right) or 'L' (left)
            multiplier: +1 or -1 (for vector math)
            angle_diff: signed angle difference in degrees
    """
    # Normalize angles to [0, 360)
    COG = COG % 360
    TWD = TWD % 360

    # Calculate angle difference [-180, 180]
    angle_diff = ((COG - TWD + 180) % 360) - 180

    # Upwind/Downwind: upwind if |angle| < 90, downwind if |angle| > 90
    rel_wind = 'UW' if abs(angle_diff) < 90 else 'DW'

    # Tack: right if angle_diff > 0, left if angle_diff < 0
    tack = 'R' if angle_diff > 0 else 'L'

    # Multiplier: +1 for right, -1 for left
    multiplier = 1 if tack == 'R' else -1

    return rel_wind, tack, multiplier, angle_diff


lambda_rel_wind = lambda row : pd.Series(detrmine_DW_UW_R_L(row['COG - Course over Ground'],row['TWD - True Wind Direction']))


# In[273]:


sailmon_cut[['rel_wind - UW/DW', 'tack - R/L','multiply const', 'angle diff [deg]']] = sailmon_cut.apply(lambda_rel_wind, axis=1)

sailmon_cut


# In[274]:


sailmon_cut = sailmon_cut.copy()
sailmon_cut['COG_diff'] = abs(sailmon_cut['COG - Course over Ground'].diff())
sailmon_cut


# In[275]:


sailmon_cut['angle_diff_diff'] = sailmon_cut['angle diff [deg]'].diff()
sailmon_cut


# In[276]:


sailmon_cut['SOG_diff'] = sailmon_cut['SOG - Speed over Ground'].diff()


# In[277]:


sailmon_cut['cog_smooth'] = sailmon_cut['COG - Course over Ground'].rolling(window=15, center=True, min_periods=1).mean()


# In[278]:


sailmon_cut


# In[279]:


sailmon_cut['COG - Course over Ground'].describe()


# In[280]:


# Identify transition points in 'tack - R/L'
sailmon_cut['transition'] = sailmon_cut['tack - R/L'].ne(sailmon_cut['tack - R/L'].shift())

# Filter points before transitions
transition_points = sailmon_cut[sailmon_cut['transition']].copy()


# In[281]:


transition_points


# In[282]:


transition_points['tack'] =  (
    transition_points['tack - R/L'].shift(-1) + " to " + transition_points['tack - R/L']
)


# In[283]:


transition_points


# In[284]:


# Define a function to classify transition types
def classify_transition(transition):
    if transition == "R to L":
        return "Starboard"
    elif transition == "L to R":
        return "Port"
    else:
        return "Other"


# In[285]:


# Apply the function to classify transitions
transition_points['turn_tack'] = transition_points['tack'].apply(classify_transition)


# In[286]:


transition_points


# In[287]:


# Determine the turn type based on the value in 'rel_wind - UW/DW'
transition_points['turn_type'] = np.where(
    transition_points['rel_wind - UW/DW'] == "DW", "Jibe",
    np.where(transition_points['rel_wind - UW/DW'] == "UW", "Tack", "Other")
)


# In[288]:


transition_points


# # Identify transition points in 'tack - R/L'
# sailmon_cut['transition'] = sailmon_cut['tack - R/L'].ne(sailmon_cut['tack - R/L'].shift())
# 
# # Filter points before transitions
# transition_points = sailmon_cut[sailmon_cut['transition']].copy()
# 
# # Determine transition type
# transition_points['transition_type'] = (
#     transition_points['tack - R/L'].shift(-1) + " to " + transition_points['tack - R/L']
# )
# transition_points.dropna(inplace=True)

# transition = np.where(sailmon_cut['transition']==True)

# In[290]:


transition_points_velocity_filtered = transition_points[transition_points['sog_smooth[knots]'] >= 8]
transition_points_velocity_filtered


# In[293]:



df_turns = transition_points_velocity_filtered.copy()

df_turns['peak_time'] = pd.to_datetime(df_turns['servertime'])

df_turns['servertime_unix'] = df_turns['servertime'].astype('int64') // 10**9

df_turns['start_time'] = df_turns['peak_time'] - timedelta(seconds=5)
df_turns['end_time'] = df_turns['peak_time'] + timedelta(seconds=5)

df_turns['turn_index'] = range(1, len(df_turns) + 1)

df_summary = df_turns[['turn_index', 'turn_type', 'start_time', 'peak_time', 'end_time']]

df_summary


# In[295]:


#df_summary.to_csv('output_turn_datetime.csv', index=False)


# In[297]:


df_turns_unix = df_summary.copy()
for col in ['start_time', 'peak_time', 'end_time']:
    df_turns_unix[col] = df_turns_unix[col].apply(lambda x: x.timestamp())


# In[299]:


# Filter for rows where 'turn_type' is 'Tack'
tack_rows = transition_points[transition_points['turn_type'] == "Tack"]

# Select only the columns 'time', 'turn tack', and 'turn_type'
tack_filtered = tack_rows[['time', 'turn_type','turn_tack']]

# Display the result
#print(tack_filtered)


# In[301]:


def angle_between_lines(m1, m2):
    angle_rad = np.arctan(np.abs((m1 - m2) / (1 + m1 * m2)))
    return np.degrees(angle_rad)

# התאמת קו ישר עם RANSAC
def fit_line_ransac(x, y):
    model = RANSACRegressor()
    x = x.reshape(-1, 1)
    model.fit(x, y)
    return model.estimator_.coef_[0]  # השיפוע של הקו


# In[302]:


traj_df = sailmon_cut


# In[303]:


# ---------- הגדרות ----------

window = 10  # מספר נקודות לפני ואחרי הפנייה
min_angle = 33
max_angle = 110
df_turns['df_index'] = df_turns.index

# df: טבלת נתיב ראשית (מכילה latitude, longitude)
# turns_df: טבלה עם peak_time של הפניות
# נניח שכבר הפכנו את עמודות הזמנים לתאריכים:


# In[304]:


# ---------- חישוב ואימות פניות ----------

confirmed_turns = []
rejected_turns = []

for _, row in df_turns.iterrows():
    turn_index = row['turn_index']
    idx = row['df_index']

    if idx - window < 0 or idx + window >= len(traj_df):
        rejected_turns.append((idx, "Out of bounds"))
        continue

    # קטעים לפני ואחרי הפנייה
    seg1 = traj_df.iloc[idx - window:idx]
    seg2 = traj_df.iloc[idx + 1:idx + 1 + window]

    # שימוש ב-longitude כ-x ו-latitude כ-y
    x1, y1 = seg1['longitude'].values, seg1['latitude'].values
    x2, y2 = seg2['longitude'].values, seg2['latitude'].values

    try:
        m1 = fit_line_ransac(x1, y1)
        m2 = fit_line_ransac(x2, y2)
        angle = angle_between_lines(m1, m2)
    except Exception as e:
        rejected_turns.append((turn_index, idx, f"Fitting error: {str(e)}"))
        continue

    if angle < min_angle:
        rejected_turns.append((turn_index, idx, f"Angle too small: {angle:.2f}°"))
    elif angle > max_angle:
        rejected_turns.append((turn_index, idx, f"Angle too large: {angle:.2f}°"))
    else:
        confirmed_turns.append((turn_index, idx, angle))


# In[305]:


# ---------- הצגת תוצאה עם המספור המקורי ----------
confirmed_df = pd.DataFrame(confirmed_turns, columns=["turn_index", "df_index", "Angle (degrees)"])
rejected_df = pd.DataFrame(rejected_turns, columns=["turn_index", "df_index", "Reason for rejection"])


# In[308]:


df_turns['valid_turn'] = True
rejected_turn_indexes = rejected_df['turn_index'].unique()
df_turns.loc[df_turns['turn_index'].isin(rejected_turn_indexes),'valid_turn'] = False


# In[309]:


inlier_turns = df_turns[df_turns['valid_turn']]


# In[312]:


def fit_line_and_error(x, y):
    """מתאים קו בעזרת RANSAC ומחזיר גם שגיאה (MSE) וגם השיפוע"""
    model = RANSACRegressor()
    model.fit(x.reshape(-1, 1), y)
    y_pred = model.predict(x.reshape(-1, 1))
    error = mean_squared_error(y, y_pred)
    slope = model.estimator_.coef_[0]
    return error, slope

def find_straight_segment(traj_df, center_idx, direction='backward', max_window=30, min_samples=5, error_threshold=1e-6):
    """
    הולך אחורה/קדימה מהשיא עד שהשגיאה של ההתאמה לקו יורדת מתחת לסף
    מחזיר את האינדקס שבו מתחיל (או מסתיים) אזור ישר
    """
    if direction == 'backward':
        indices = range(center_idx - min_samples, max(center_idx - max_window, 0), -1)
    else:  # forward
        indices = range(center_idx + min_samples, min(center_idx + max_window, len(traj_df)))

    for i in indices:
        if direction == 'backward':
            x = traj_df['longitude'].iloc[i:center_idx].values
            y = traj_df['latitude'].iloc[i:center_idx].values
            start_idx = i
        else:
            x = traj_df['longitude'].iloc[center_idx:i].values
            y = traj_df['latitude'].iloc[center_idx:i].values
            start_idx = i

        if len(x) < min_samples:
            continue

        error, slope = fit_line_and_error(np.array(x), np.array(y))
        if error < error_threshold:
            return start_idx

    return None

def apply_turn_detection_by_good_fit(inlier_turns, traj_df, max_window=30, error_threshold=1e-6):
    """
    מפעיל על כל שיא פנייה:
    - מוצא התחלה (אחורה) וסיום (קדימה) לפי שיפור בהתאמה לקו ישר
    """
    results = []

    for _, row in inlier_turns.iterrows():
        center_idx = row['df_index']

        if center_idx < max_window or center_idx >= len(traj_df) - max_window:
            continue  # לא מספיק נתונים סביב הפנייה

        start_idx = find_straight_segment(traj_df, center_idx, direction='backward', max_window=max_window, error_threshold=error_threshold)
        end_idx = find_straight_segment(traj_df, center_idx, direction='forward', max_window=max_window, error_threshold=error_threshold)

        if start_idx is not None and end_idx is not None:
            results.append({
                'turn_index': row['turn_index'],
                'df_index': center_idx,
                'start_idx': start_idx,
                'end_idx': end_idx
            })

    return pd.DataFrame(results)


# In[313]:


new_turn_boundaries_df = apply_turn_detection_by_good_fit(
    inlier_turns=inlier_turns,  # זה ה-DataFrame עם הפניות שכבר סיננת
    traj_df=traj_df,            # זה ה-DataFrame עם המסלול המלא
    max_window=10,
    error_threshold=1e-7
)


# In[314]:


new_turn_boundaries_df['turn_index_update'] = range(1, len(new_turn_boundaries_df) + 1)
new_turn_boundaries_df


# In[315]:


def create_turns_summary_table(turn_boundaries_df, inlier_turns, traj_df):
    """
    מחזיר DataFrame בפורמט:
    "",turn_index,turn_type,start_time,peak_time,end_time
    כשהאינדקס הוא df_index (peak_index)
    והזמנים הם datetime מלא
    """

    # המרה ל-datetime אם עדיין לא בוצעה
    traj_df['timestamp'] = pd.to_datetime(traj_df['time'])

    # מיזוג עם סוג הפנייה
    merged = turn_boundaries_df.merge(
        inlier_turns[['turn_index', 'turn_type']], on='turn_index', how='left'
    )

    # שליפת זמני התחלה, שיא, וסיום לפי האינדקסים
    merged['start_time'] = merged['start_idx'].apply(lambda idx: traj_df.loc[idx, 'timestamp'])
    merged['peak_time'] = merged['df_index'].apply(lambda idx: traj_df.loc[idx, 'timestamp'])
    merged['end_time'] = merged['end_idx'].apply(lambda idx: traj_df.loc[idx, 'timestamp'])

    # הגדרת df_index כאינדקס
    merged.set_index('df_index', inplace=True)

    # בחירת עמודות מסודרת
    summary_df = merged[['turn_index', 'turn_type', 'start_time', 'peak_time', 'end_time','turn_index_update']].copy()

    # שם עמודת אינדקס - "" כמו בדוגמה שלך
    summary_df.index.name = '""'

    return summary_df


# In[316]:


# איפוס אינדקסים לפי הסדר החדש אחרי סינון או חישוב מחדש
new_turn_boundaries_df = new_turn_boundaries_df.copy()
new_turn_boundaries_df['turn_index_update'] = range(1, len(new_turn_boundaries_df) + 1)

# התאמה של inlier_turns לאינדקסים החדשים לפי turn_index המקורי
# כאן נניח ש-new_turn_boundaries_df הגיע מפילטר על inlier_turns
# לכן נעשה מיזוג ביניהם
inlier_turns_updated = inlier_turns.merge(
    new_turn_boundaries_df[['turn_index', 'turn_index_update']],
    on='turn_index',
    how='inner'  # רק מי שעבר את הפילטר
)


# In[317]:


summary_df = create_turns_summary_table(new_turn_boundaries_df, inlier_turns, traj_df)
#summary_df.to_csv('turns_summary.csv')
summary_df = summary_df.reset_index(drop=True)


# In[321]:


def create_turns_summary_table(turn_boundaries_df, inlier_turns, traj_df):
    """
    מחזיר DataFrame בפורמט:
    turn_index, turn_type, start_time, peak_time, end_time
    """
    # לוודא שהזמנים בפורמט datetime
    traj_df['timestamp'] = pd.to_datetime(traj_df['time'])

    # מיזוג עם סוג הפנייה (Jibe או Tack)
    merged = turn_boundaries_df.merge(inlier_turns[['turn_index', 'turn_type']], on='turn_index', how='left')
    #print("merged",merged)
    # חישוב זמנים לפי אינדקסים
    merged['start_time'] = merged['start_idx'].apply(lambda idx: traj_df.loc[idx, 'timestamp'])
    merged['peak_time'] = merged['df_index'].apply(lambda idx: traj_df.loc[idx, 'timestamp'])
    merged['end_time'] = merged['end_idx'].apply(lambda idx: traj_df.loc[idx, 'timestamp'])

    # סידור העמודות
    summary_df = merged[['turn_type', 'start_time', 'peak_time', 'end_time']].copy()

    return summary_df


summary_df = create_turns_summary_table(new_turn_boundaries_df, inlier_turns, traj_df)
summary_df.to_csv("turns.csv", index=True)
summary_df['turn_index'] = range(1, len(summary_df) + 1)