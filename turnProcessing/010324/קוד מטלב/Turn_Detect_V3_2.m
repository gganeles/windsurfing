clear
clc
cd DATA_for_perf\

% Load the data
data = readtable('2024-03-01-sailmon-1 TomR.csv');

%% load performance data:
cd ..
experimentnumber = input('Please enter the experiment number: ');
filename1 = sprintf('Experiment%d_TurnPerfData.mat', experimentnumber);
load(filename1)

%%

% Initialize a variable to store the time of the first non-NaN value
first_non_nan_time = NaT; % NaT (Not-a-Time) is used as an empty datetime placeholder

% Loop through the latitude data to find the first non-NaN value
for i = 1:length(Sailmon_1_I.S1I_Latitude)
    if ~isnan(Sailmon_1_I.S1I_Latitude(i))
        first_non_nan_time = Sailmon_1_I.CORI_TimeVector(i);
        break;
    end
end

data.time.TimeZone = first_non_nan_time.TimeZone;

endtime=Sailmon_1_I.CORI_TimeVector(end);
%set manual starttime and endtime if you want@@@@@@@@@@@@@

time_diff_startind = abs(data.time - first_non_nan_time);
time_diff_endind = abs(data.time - endtime);
[~, indexstart] = min(time_diff_startind);
[~, index_end] = min(time_diff_endind);

%%

% Extract necessary columns within the specified indices
time = data.time(indexstart:index_end);
SOG = data.SOG_SpeedOverGround(indexstart:index_end);
COG = data.COG_CourseOverGround(indexstart:index_end);
latitude = data.latitude(indexstart:index_end);
longitude = data.longitude(indexstart:index_end);

% Define thresholds and parameters
Vt = 4; % Speed threshold in m/s
min_turn_time = 5; % Minimum turn time in seconds
max_turn_time = 15; % Maximum turn time in seconds
min_angle_change = 60; % Minimum angle change in degrees
max_angle_change = 120; % Maximum angle change in degrees
T1 = seconds(20); % Time to skip forward after detecting a turn

% Preprocess COG to handle wrap-around at 0/360 degrees
COG_unwrapped = unwrap(deg2rad(COG)); % Convert to radians and unwrap
COG_unwrapped = rad2deg(COG_unwrapped); % Convert back to degrees

% Initialize the turns table with correct empty arrays
turns = table([], [], [], [], [], [], [], [], [], 'VariableNames', {'StartTime', 'EndTime', 'StartCOG', 'EndCOG', 'AngleChange', 'StartLat', 'StartLon', 'EndLat', 'EndLon'});

% Identify potential turns
i = 1;
while i <= length(COG_unwrapped) - min_turn_time
    if SOG(i) >= Vt
        for j = i+min_turn_time:i+max_turn_time
            if j <= length(COG_unwrapped)
                angle_change = abs(COG_unwrapped(j) - COG_unwrapped(i));
                if angle_change >= min_angle_change && angle_change <= max_angle_change
                    newTurn = {time(i), time(j), COG(i), COG(j), angle_change, latitude(i), longitude(i), latitude(j), longitude(j)};
                    turns = [turns; newTurn]; %#ok<AGROW>
                    % Skip forward by T1 seconds
                    i = find(time >= time(i) + T1, 1);
                    break;
                end
            end
        end
    end
    i = i + 1;
end


%%
% Initialize a new column in the turns table for peak points
turns.PeakLat = NaN(height(turns), 1);
turns.PeakLon = NaN(height(turns), 1);

% Initialize a struct array to store the detected turn peaks
turnPeaks = struct('Time_UTC', {}, 'Latitude', {}, 'Longitude', {});

% Parameters for peak detection
search_window = 20; % Number of points before and after the start point to consider

% Loop over each detected turn
for k = 1:height(turns)
    % Get the start index for the current turn
    start_time = turns.StartTime(k);
    start_idx = find(time == start_time);
    
    % Define the search segment around the start index
    start_segment_idx = max(1, start_idx);
    end_segment_idx = min(length(latitude), start_idx + search_window);
    
    % Extract the lat/lon data in the search segment
    lat_segment = latitude(start_segment_idx:end_segment_idx);
    lon_segment = longitude(start_segment_idx:end_segment_idx);
    time_segment = time(start_segment_idx:end_segment_idx);
    
    % Calculate the curvature in this segment
    curvature = NaN(length(lat_segment) - 2, 1);
    for n = 2:length(lat_segment)-1
        % Points for curvature calculation
        x1 = lon_segment(n-1);
        y1 = lat_segment(n-1);
        x2 = lon_segment(n);
        y2 = lat_segment(n);
        x3 = lon_segment(n+1);
        y3 = lat_segment(n+1);
        
        % Calculate the angle change between consecutive segments
        angle1 = atan2(y2 - y1, x2 - x1);
        angle2 = atan2(y3 - y2, x3 - x2);
        curvature(n-1) = abs(angle2 - angle1);
    end
    
    % Find the point with maximum curvature
    [~, peak_idx] = max(curvature);
    peak_idx = peak_idx + 1; % Adjust index for the original segment
    
    % Store the peak time, lat, and lon in the struct array
    PTimeUTC(k) = time_segment(peak_idx);% TIME in UTC @@@@@@@@@
    PLat(k) = lat_segment(peak_idx);
    PLong(k) = lon_segment(peak_idx);
end

PTimeIsr=PTimeUTC;

currentTimeZone = PTimeIsr.TimeZone;
disp(['Current Time Zone: ', currentTimeZone]);
% Set the time zone to Israel Standard Time (Asia/Jerusalem)
PTimeIsr.TimeZone = 'Asia/Jerusalem';
% Display in the desired format
PTimeIsr.Format = 'yyyy-MM-dd HH:mm:ss.SSS';

PTimeUTC=PTimeUTC';
PTimeIsr=PTimeIsr';
PLat=PLat';
PLong=PLong';

turnPeaks = table(PTimeUTC,PTimeIsr,PLat,PLong);
%%

% Plot the GPS data with detected turns and their peaks
figure;
geoplot(latitude, longitude, 'b');
hold on;

% Plot peak points using the turnPeaks struct
if ~isempty(turnPeaks)
    % Extract latitudes and longitudes from the turnPeaks struct
    peakLatitudes = [turnPeaks.PLat];
    peakLongitudes = [turnPeaks.PLong];
    
    % Plot the peak points
    geoplot(peakLatitudes, peakLongitudes, 'mo', 'MarkerSize', 8, 'DisplayName', 'Turn Peak');
    
    % Annotate each peak with its index
    for idx = 1:height(turnPeaks) % Use height instead of length
        text(peakLatitudes(idx), peakLongitudes(idx), num2str(idx), ...
             'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 8, 'Color', 'black');
    end
end

title('GPS Track with Turn Detection and Peaks');
legend('GPS Track', 'Turn Peak');
hold off;


%%
% Save the turnPeaks struct array to a .mat file
filename2 = sprintf('detected_turnPeaks%d.mat', experimentnumber);
save(filename2, 'turnPeaks');

% Save the table to a CSV filec
writetable(turnPeaks, 'detected_turnspeaks.csv');


