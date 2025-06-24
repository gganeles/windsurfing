import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
from geopy.distance import geodesic
import scipy.signal as sig




def merge_sorted_lists(list1, list2):
    # Initialize pointers for both lists
    i, j = 0, 0
    merged_list = []

    # Traverse both lists and append the smallest element
    while i < len(list1) and j < len(list2):
        if list1[i] <= list2[j]:
            merged_list.append(list1[i])
            i += 1
        else:
            merged_list.append(list2[j])
            j += 1

    # Append any remaining elements from list1
    while i < len(list1):
        merged_list.append(list1[i])
        i += 1

    # Append any remaining elements from list2
    while j < len(list2):
        merged_list.append(list2[j])
        j += 1

    return merged_list


def turnEffectiveness(timestamps: list, gps: pd.DataFrame, exiting = False, plot=False):
    # Get the GPS data for the given timestamp
    effectivenessVector = []
    
    fc = 0.1
    a, b = sig.butter(2, fc, 'low', analog=False,fs=1)
    smoothCOG = sig.filtfilt(a, b, gps["COG - Course over Ground"])
    smoothVel = sig.filtfilt(a, b, gps["SOG - Speed over Ground"])
    TWD_AVG = 90 #gps["TWD - True Wind Direction"].mean()
    TWA = TWD_AVG-smoothCOG
    
    smoothCOGdiff = np.diff(smoothCOG)
    
    # find peaks
    peaks, _ = sig.find_peaks(smoothCOG)
    
    valleys, _ = sig.find_peaks(-smoothCOG)
    
    peaks = np.array(merge_sorted_lists(peaks, valleys))
    
    timeVec = datetimer(gps["time"])
    
    
    #TWA = TWD_AVG-smoothCOG

    vmg = smoothVel * np.cos(np.radians(TWA))

    vmgList = []
    for i, timestamp in enumerate(timestamps):
        matchingTimstep = np.where((timeVec >= timestamp))
        idx = matchingTimstep[0][0]
        
        peaks_before_timestamp = peaks[peaks < idx][-3:]
        peaks_after_timestamp = peaks[peaks > idx][:2]
        
        vmgList.extend([*vmg[peaks_before_timestamp],*vmg[peaks_after_timestamp]])
    
        
    for i, timestamp in enumerate(timestamps):
        matchingTimstep = np.where((timeVec >= timestamp))
        idx = matchingTimstep[0][0]
        
        peaks_before_timestamp = peaks[peaks < idx][-3:]
        peaks_after_timestamp = peaks[peaks > idx][:2]

        if not exiting:
            sectionInds = range(peaks_before_timestamp[0], peaks_after_timestamp[1]+1)

            # Calculate the distance travelled over the course of the section
            start_coords = (gps["latitude"].iloc[sectionInds[0]], gps["longitude"].iloc[sectionInds[0]])
            end_coords = (gps["latitude"].iloc[sectionInds[-1]], gps["longitude"].iloc[sectionInds[-1]])
            distance_travelled = geodesic(start_coords, end_coords).meters
            # Calculate the angle between the geodesic and TWD
            geodesic_angle = np.arctan2(
                end_coords[1] - start_coords[1],
                end_coords[0] - start_coords[0]
            )
            angle_diff = np.radians(TWD_AVG) - geodesic_angle
            print(f"Geodesic angle: {geodesic_angle}")
            print(f"TWD: {np.radians(TWD_AVG)}")
            print(f"Angle diff: {angle_diff}")
            cos_angle_diff = np.cos(angle_diff)


            projected_distance = distance_travelled * cos_angle_diff

            q3vel = np.quantile(vmgList, 0.75)
            virtualDistance = q3vel*(timeVec[peaks_after_timestamp[-1]+1]-timeVec[peaks_before_timestamp[0]]).total_seconds()
            effectivenessVector.append((projected_distance)/virtualDistance)
            
        else:
            sectionInds = range(idx, peaks_after_timestamp[1]+1)
            
            # Calculate the distance travelled over the course of the section
            start_coords = (gps["latitude"].iloc[sectionInds[0]], gps["longitude"].iloc[sectionInds[0]])
            end_coords = (gps["latitude"].iloc[sectionInds[-1]], gps["longitude"].iloc[sectionInds[-1]])
            distance_travelled = geodesic(start_coords, end_coords).meters
            # Calculate the angle between the geodesic and TWD
            geodesic_angle = np.arctan2(
                end_coords[1] - start_coords[1],
                end_coords[0] - start_coords[0]
            )
            
            # Plot the coordinates (latitude, longitude) and TWD direction

            
            angle_diff = np.radians(TWD_AVG) - geodesic_angle
            print(f"Geodesic angle: {geodesic_angle}")
            print(f"TWD: {np.radians(TWD_AVG)}")
            print(f"Angle diff: {angle_diff}")
            cos_angle_diff = np.cos(angle_diff)

            q3vel = np.quantile(vmgList, 0.75)
            projected_distance = distance_travelled * cos_angle_diff
            virtualDistance = vmg[sectionInds[-1]]*(timeVec[sectionInds[-1]]-timeVec[sectionInds[0]]).total_seconds()
            effectivenessVector.append((projected_distance)/virtualDistance)
            
        
        
        print(f"Distance travelled: {distance_travelled} meters")
        print(f"Projected distance in TWA direction: {projected_distance} meters")
        print(f"Virtual distance: {virtualDistance} meters")
        
        
        sectionTimeVec = timeVec[sectionInds]
        # find three peaks (take avg of last two) in the cog signal before turn, two after (last one)
        
        # plt.plot(np.arange(0, len(gpsSection)),twa, label="twa")    
        # plt.plot(np.arange(0, len(gpsSection)),np.cos(twa), label="cos twa")
        # plt.plot(np.arange(0, len(gpsSection)),rad_cog, label="velocity")
        # plt.plot([0, len(gpsSection)],[rad_twd,rad_twd], label="twd")
        
        if plot:
            plt.figure()
            plt.plot(gps["longitude"], gps["latitude"], label="Path")
            plt.quiver(gps["longitude"].iloc[sectionInds], gps["latitude"].iloc[sectionInds], 
                        np.sin(np.radians(TWD_AVG)), np.cos(np.radians(TWD_AVG)), 
                        color='r', scale=10, label="TWD")
            plt.scatter([start_coords[1], end_coords[1]], [start_coords[0], end_coords[0]], color='g', label="Start/End")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.legend()
            plt.title("Path and TWD Direction")
            plt.show()
        
        if plot:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(sectionTimeVec,vmg[sectionInds], label="VMG")
            plt.plot([sectionTimeVec[0], sectionTimeVec[-1]],[q3vel, q3vel], label="Q3 Velocity")
            #plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(sectionTimeVec,TWA[sectionInds], label="TWA")
            plt.scatter(timeVec[peaks_before_timestamp],TWA[peaks_before_timestamp], color="red")
            plt.scatter(timeVec[peaks_after_timestamp],TWA[peaks_after_timestamp], color="green")
            plt.plot(timeVec[idx],TWA[idx], marker="o", color="black")
            #plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
            plt.legend()
            plt.show()

        
    return effectivenessVector


output_file = "turnAnalysis.csv"
gopro_file = "DATA_for_perf\\GL_TomR_0095.csv"
sailmon_file = "DATA_for_perf\\2024-03-01-sailmon-1 TomR.csv"




goProData = pd.read_csv(gopro_file)
sailmonData = pd.read_csv(sailmon_file)
turns = pd.read_csv("DATA_for_perf\detected_turns.csv")
datetimer = np.vectorize(parser.parse)
dateStringer = np.vectorize(lambda x: x.strftime('%Y-%m-%d %H:%M:%S.%f'))


#timestamps = np.array(dateStringer((datetimer(turns["EndTime"]) - datetimer(turns["StartTime"]))/2 + datetimer(turns["StartTime"])))

data = pd.read_csv('turnPeaks.csv')
timestamps = datetimer(data["PTimeUTC"])

i = (sailmonData["TWA - True Wind Angle"].isna() != True).idxmax()

validtacks = np.array([14,15,16,18,19,20,21,27,28,29])-1

#turnEffectiveness(timestamps[validtacks], sailmonData)


outdata = pd.DataFrame()

outdata["index"] = validtacks+1
#outdata["time"] = timestamps[validtacks]

outdata["turnEff"] = turnEffectiveness(timestamps[validtacks], sailmonData)
outdata["turnEffExit"] = turnEffectiveness(timestamps[validtacks], sailmonData,True,True)
outdata.to_csv('turnEffectiveness.csv', index=False)