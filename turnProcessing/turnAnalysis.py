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


def turnEffectiveness(data: pd.DataFrame, gps: pd.DataFrame, debug=False):
	# Get the GPS data for the given timestamp
	output = pd.DataFrame(columns=["turn_index", "turnLength", "lost_distance", "effectiveness", "exit_lost_distance", "exitEffectiveness"])
	
	if type(data["peak_time"]) == str:
		timestamps = datetimer(np.array([data["peak_time"]]))
	else:
		timestamps = np.array(datetimer(data["peak_time"]))

	fc = 0.1
	a, b = sig.butter(2, fc, 'low', analog=False, fs=1)
	smoothCOG = sig.filtfilt(a, b, gps["COG - Course over Ground"])

	fc = 0.2
	a, b = sig.butter(2, fc, 'low', analog=False, fs=1)
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
	
	exitVmgList = []
	# Calculate q3vmg (75th percentile VMG) for the "exiting" section after each timestamp
	for i, timestamp in enumerate(timestamps):
		

		matchingTimstep = np.where((timeVec >= timestamp))
		idx = matchingTimstep[0][0]
		
		peaks_after_timestamp = peaks[peaks > idx][:2]
		if len(peaks_after_timestamp) < 2:
			# Not enough data to define exit section
			exitVmgList.append(np.nan)
			continue
		exitVmgList.extend([*vmg[peaks_after_timestamp]])

	
	for i, timestamp in enumerate(timestamps):
		if (type(data["turn_type"]) == str and data["turn_type"].lower() == "jibe") or (type(data["turn_type"]) == pd.Series and data["turn_type"].iloc[i].lower() == "jibe"):
			vmg = -vmg;
			#TWA = TWA+180;

		matchingTimstep = np.where((timeVec >= timestamp))
		idx = matchingTimstep[0][0]
		
		peaks_before_timestamp = peaks[peaks < idx][-3:]
		peaks_after_timestamp = peaks[peaks > idx][:2]

		
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
		#if (type(data["turn_type"]) == str and data["turn_type"].lower() == "jibe") or (type(data["turn_type"]) == pd.Series and data["turn_type"].iloc[i].lower() == "jibe"):
		if debug:
			print()
			print("Turn Statistics:")
			print(f"Geodesic angle: {geodesic_angle}")
			print(f"TWD: {np.radians(TWD_AVG)}")
			print(f"Angle diff: {angle_diff}")

		cos_angle_diff = np.abs(np.cos(angle_diff))


		projected_distance = distance_travelled * cos_angle_diff

		#q3vel = np.quantile(vmgList, 0.75)
		vel = vmg[peaks_before_timestamp[0]:peaks_before_timestamp[1]].mean()

		virtualDistance = vel*(timeVec[peaks_after_timestamp[-1]+1]-timeVec[peaks_before_timestamp[0]]).total_seconds()
		lostDistance = virtualDistance - projected_distance
		if debug:
			print(f"Distance travelled: {distance_travelled} meters")
			print(f"Projected distance in TWA direction: {projected_distance} meters")
			print(f"Virtual distance: {virtualDistance} meters")
			print(f"Lost distance: {lostDistance} meters")
			print(f"Effectiveness: {projected_distance/virtualDistance}")

		
		exitSectionInds = range(idx, peaks_after_timestamp[1]+1)
		
		# Calculate the distance travelled over the course of the section
		exit_start_coords = (gps["latitude"].iloc[exitSectionInds[0]], gps["longitude"].iloc[exitSectionInds[0]])
		distance_travelled = geodesic(exit_start_coords, end_coords).meters
		# Calculate the angle between the geodesic and TWD
		geodesic_angle = np.arctan2(
			end_coords[1] - exit_start_coords[1],
			end_coords[0] - exit_start_coords[0]
		)
		
		# Plot the coordinates (latitude, longitude) and TWD direction

		
		angle_diff = np.radians(TWD_AVG) - geodesic_angle

		if debug:
			print()
			print("Exit Statistics:")
			print(f"Geodesic angle: {geodesic_angle}")
			print(f"TWD: {np.radians(TWD_AVG)}")
			print(f"Angle diff: {angle_diff}")
		cos_angle_diff = np.cos(angle_diff)

		#q3vel = np.quantile(exitVmgList, 0.75)
		exiting_projected_distance = distance_travelled * cos_angle_diff
		exiting_virtualDistance = vmg[exitSectionInds[-1]]*(timeVec[exitSectionInds[-1]]-timeVec[exitSectionInds[0]]).total_seconds()
		exitingLostDistance = exiting_virtualDistance - exiting_projected_distance
		
		
		output.loc[i] = [i, (timeVec[peaks_after_timestamp[-1]]-timeVec[peaks_before_timestamp[1]]).total_seconds(), lostDistance, projected_distance/virtualDistance, exitingLostDistance, exiting_projected_distance/exiting_virtualDistance]
	

		if debug:

			print(f"Projected Exiting distance in TWA direction: {exiting_projected_distance} meters")
			print(f"Virtual Exiting distance: {exiting_virtualDistance} meters")
			print(f"Exiting Lost distance: {exitingLostDistance} meters")
			print(f"Exiting Effectiveness: {exiting_projected_distance/exiting_virtualDistance}")
		
		
		sectionTimeVec = timeVec[sectionInds]
		# find three peaks (take avg of last two) in the cog signal before turn, two after (last one)
		
		# plt.plot(np.arange(0, len(gpsSection)),twa, label="twa")    
		# plt.plot(np.arange(0, len(gpsSection)),np.cos(twa), label="cos twa")
		# plt.plot(np.arange(0, len(gpsSection)),rad_cog, label="velocity")
		# plt.plot([0, len(gpsSection)],[rad_twd,rad_twd], label="twd")
		
		if debug:
			from matplotlib.ticker import MaxNLocator

			plt.figure()
			plt.suptitle("Turn Number: "+str(data.index[i]))
						

			plt.subplot(2,1,1)
			plt.plot(gps["longitude"], gps["latitude"], label="Full path")
			plt.quiver(gps["longitude"].iloc[sectionInds[::5]], gps["latitude"].iloc[sectionInds[::5]], 
			            np.sin(np.radians(TWD_AVG)), np.cos(np.radians(TWD_AVG)), 
			            color='r', scale=19, label="True Wind Direction during turn", width=0.003, headwidth=3, headlength=5)
			plt.scatter([start_coords[1]], [start_coords[0]],label="Start", color="r",zorder=10)
			plt.scatter(gps["longitude"].iloc[idx],gps["latitude"].iloc[idx], label="Peak", color="black",zorder=10)
			plt.scatter([end_coords[1]], [end_coords[0]],label="End", color="g",zorder=10)
			plt.xlabel("Longitude")
			plt.ylabel("Latitude")
			#plt.legend()
			plt.title("Path and TWD Direction")
			#plt.show()

			pltXvec = np.arange(sectionInds[0]-10,sectionInds[-1]+10)
			sectionInds = pltXvec
			sectionTimeVec=timeVec[pltXvec]

			plt.subplot(4,1,3)
			plt.plot(sectionTimeVec,TWA[sectionInds], label="True wind angle")
			plt.scatter(timeVec[peaks_before_timestamp],TWA[peaks_before_timestamp], color="red", label="Extrema before turn")
			plt.scatter(timeVec[peaks_after_timestamp],TWA[peaks_after_timestamp], color="green", label="Extrema after turn")
			plt.plot(timeVec[idx],TWA[idx], marker="o", color="black")
			plt.ylabel("TWA (deg)")
			plt.xlabel("UTC DateTime")
			plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
			#plt.legend()
			
			#plt.show()
			#plt.figure()
			
			#plt.title("Turn Number: "+str(data.index[i]), fontsize=18)
			plt.subplot(4,1,4)
			plt.plot(sectionTimeVec,vmg[sectionInds], label="Velocity made good",lw=5)
			plt.plot([sectionTimeVec[0], sectionTimeVec[-1]],[vel, vel], label="Avg entrance velocity",lw=5)
			plt.scatter(timeVec[peaks_before_timestamp],vmg[peaks_before_timestamp], color="red",zorder=10,s=50)
			plt.scatter(timeVec[peaks_after_timestamp],vmg[peaks_after_timestamp], color="green",zorder=10,s=50)
			#plot rectangle representing projected distance
			plt.fill([sectionTimeVec[0], sectionTimeVec[-1], sectionTimeVec[-1], sectionTimeVec[0]],
					 [0, 0, vel, vel], color="orange", alpha=0.2)
			plt.fill_between(sectionTimeVec, 0, vmg[sectionInds], color="blue", alpha=0.2)
			plt.ylabel("Velocity (m/s)")#, fontsize=18)
			plt.xlabel("UTC DateTime")#, fontsize=18)
			plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
			#plt.legend()#fontsize=18)
			plt.subplots_adjust(hspace=0.4)
			plt.show()



		
	return output


datetimer = np.vectorize(parser.parse)

if __name__ == "__main__":
	import os
	params_txt = "paramsForTurnAnalysis.txt"
	if not os.path.exists(params_txt):
		print("Error: paramsForTurnAnalysis.txt file not found")
		exit()
	with open(params_txt, 'r', encoding='utf-16') as file:
		for line in file:
			if "Turn Numbers" in line:
				if line.split(": ")[1].strip() == "all":
					turn_numbers = "all"
				else:
					turn_numbers = np.array(list(map(int, line.split(": ")[1].strip().split(","))))
			if "Type" in line:
				turn_type = line.split(": ")[1].strip()
			if "SailmonCsvFileName" in line:
				sailmon_file = line.split(": ")[1].strip()
			if "OutputFileName" in line:
				output_file = line.split(": ")[1].strip()
			if "TurnPeaksFileName" in line:
				turn_peaks_file = line.split(": ")[1].strip()
			if "sort_by" in line:
				sort_by = line.split(": ")[1].strip()
		if not sailmon_file or not turn_peaks_file or not output_file or not np.any(turn_numbers):
			print("Error: Missing required parameters in paramsForTurnAnalysis.txt")
			exit()



	sailmonData = pd.read_csv(sailmon_file)
	data = pd.read_csv(turn_peaks_file)



   

	#i = (sailmonData["TWA - True Wind Angle"].isna() != True).idxmax()

	if type(turn_numbers) == str:
		if turn_numbers == "all" and turn_type.lower() == "all":
			validtacks = np.arange(data.shape[0])
		else:
			validtacks = data[data["turn_type"].str.lower() == turn_type.lower()].index
	else:
		validtacks = np.array(turn_numbers)


	#validtacks = np.array([21])
	#turnEffectiveness(timestamps[validtacks], sailmonData)


	outdata = pd.DataFrame()


	#outdata["time"] = timestamps[validtacks]

	outdata = turnEffectiveness(data.iloc[validtacks], sailmonData, debug=False)
	outdata["turn_index"] = validtacks

	if sort_by == "turn_index":
		outdata.sort_values(by=sort_by, ascending=True, inplace=True)
	else:
		outdata.sort_values(by=sort_by, ascending=False, inplace=True)
	outdata.to_csv(output_file, index=False)