#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <regex>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <unordered_map>

// Libraries for numerical computing and signal processing
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

// For GPMF parsing
#include "gpmf-parser/GPMF_parser.h"
#include "gpmf-parser/GPMF_utils.h"

// For FFmpeg integration
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
}

namespace fs = std::filesystem;

// Parameter struct (equivalent to Python params dict)
struct Parameters {
    double cutOff = 35.0;
    std::vector<double> passBand = {0.001, 0.5};
    double fs = 201.0;
    int kernel_n = 6000;
    double pumpThresh = 1.4;
    bool plots = false;
};

// Pump data structure
struct Pump {
    int startInd;
    int endInd;
    std::string start_str;
    std::string end_str;
    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point end;
};

// Function declarations
std::string getResourcePath(const std::string& relativePath);
Eigen::VectorXd windowRms(const Eigen::VectorXd& a, int windowSize);
std::map<std::string, std::vector<std::string>> listAndSortFiles(const std::string& directory);
std::vector<Pump> extractPumps(const Eigen::VectorXd& data, double pumpThresh);
void refinePumps(const Eigen::VectorXd& data, std::vector<Pump>& pumpList);
void addPumpMetaData(std::vector<Pump>& pumps, const std::vector<std::chrono::system_clock::time_point>& timevec, int firstTimeIndex);
std::string vidTimeStamp(const std::chrono::system_clock::time_point& datetime1, const std::chrono::system_clock::time_point& datetime2);
int hhmmssToSeconds(const std::string& hhmmss);
Eigen::VectorXd plotNormedConvolution(const Eigen::VectorXd& x, const Eigen::VectorXd& kernel, 
                                      const std::vector<std::chrono::system_clock::time_point>& timevec, 
                                      const std::string& label, bool plots);
std::vector<Pump> mainPart(const std::vector<std::string>& filenames, const std::string& filename, 
                          const Parameters& params, std::vector<std::chrono::system_clock::time_point>& timevec);
void createSupercut(const std::vector<std::string>& videoUrls, const std::vector<Pump>& timestampPairs, 
                   const std::vector<std::chrono::system_clock::time_point>& videoTimestampVector, 
                   const std::string& outputFile = "supercut.mp4", int timeOffset = 0);

// Butterworth filter implementation
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> butter(int n, double Wn, const std::string& btype = "low", double fs = 2.0) {
    // This is a simplified butter implementation - in production code you'd use a DSP library
    Eigen::MatrixXd b = Eigen::MatrixXd::Ones(1, n+1);
    Eigen::MatrixXd a = Eigen::MatrixXd::Ones(1, n+1);
    
    // Apply normalization based on filter type
    if (btype == "low") {
        double omega = 2.0 * M_PI * Wn / fs;
        double c = 1.0 / tan(omega / 2.0);
        double scale = pow(c, n);
        
        for (int i = 0; i <= n; i++) {
            b(0, i) = scale * binomialCoeff(n, i);
            a(0, i) = scale * binomialCoeff(n, i) * pow(-1.0, i);
        }
    }
    // Add other filter types implementation here...
    
    return {b, a};
}

// Helper function for butterworth filter
int binomialCoeff(int n, int k) {
    if (k == 0 || k == n) return 1;
    return binomialCoeff(n-1, k-1) + binomialCoeff(n-1, k);
}

// filtfilt implementation (zero-phase filtering)
Eigen::VectorXd filtfilt(const Eigen::MatrixXd& b, const Eigen::MatrixXd& a, const Eigen::VectorXd& x) {
    // Forward filtering
    Eigen::VectorXd y = filter(b, a, x);
    
    // Reverse the filtered signal
    Eigen::VectorXd y_reversed = y.reverse();
    
    // Filter again
    Eigen::VectorXd z = filter(b, a, y_reversed);
    
    // Reverse again to get the zero-phase filtered signal
    return z.reverse();
}

// Simple filter implementation
Eigen::VectorXd filter(const Eigen::MatrixXd& b, const Eigen::MatrixXd& a, const Eigen::VectorXd& x) {
    int n = x.size();
    int nb = b.cols();
    int na = a.cols();
    
    Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
    
    for (int i = 0; i < n; i++) {
        // Apply b coefficients (feedforward)
        for (int j = 0; j < nb; j++) {
            if (i - j >= 0) {
                y(i) += b(0, j) * x(i - j);
            }
        }
        
        // Apply a coefficients (feedback)
        for (int j = 1; j < na; j++) {
            if (i - j >= 0) {
                y(i) -= a(0, j) * y(i - j);
            }
        }
        
        // Normalize by a0
        y(i) /= a(0, 0);
    }
    
    return y;
}

// Convolution implementation
Eigen::VectorXd convolve(const Eigen::VectorXd& x, const Eigen::VectorXd& kernel, const std::string& mode = "same") {
    int nx = x.size();
    int nk = kernel.size();
    int n;
    
    if (mode == "full") {
        n = nx + nk - 1;
    } else if (mode == "same") {
        n = nx;
    } else { // "valid"
        n = nx - nk + 1;
    }
    
    Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
    
    for (int i = 0; i < n; i++) {
        int jmin = (i >= nk - 1) ? i - (nk - 1) : 0;
        int jmax = (i < nx - 1) ? i : nx - 1;
        
        for (int j = jmin; j <= jmax; j++) {
            if (i - j < nk) {
                result(i) += x(j) * kernel(i - j);
            }
        }
    }
    
    return result;
}

// Extract GPMF blocks function
std::vector<GPMF_stream> extractBlocks(const std::vector<uint8_t>& stream, const std::string& fourcc) {
    std::vector<GPMF_stream> blocks;
    // Implementation would depend on GPMF parser library
    // This is a placeholder for the actual implementation
    return blocks;
}

// Parse GPMF block
std::unordered_map<std::string, Eigen::MatrixXd> pythonifyBlock(const GPMF_stream& block, const std::string& fourcc) {
    std::unordered_map<std::string, Eigen::MatrixXd> result;
    // Implementation would depend on GPMF parser library
    // This is a placeholder for the actual implementation
    return result;
}

// Get resource path
std::string getResourcePath(const std::string& relativePath) {
    // In a real implementation, this would handle paths when running as an executable
    return relativePath;
}

// Window RMS calculation
Eigen::VectorXd windowRms(const Eigen::VectorXd& a, int windowSize) {
    Eigen::VectorXd a2 = a.array().square();
    Eigen::VectorXd window = Eigen::VectorXd::Ones(windowSize) / static_cast<double>(windowSize);
    return convolve(a2, window, "same").array().sqrt();
}

// List and sort files
std::map<std::string, std::vector<std::string>> listAndSortFiles(const std::string& directory) {
    std::map<std::string, std::vector<std::string>> filesDict;
    std::regex pattern("G.(\\d{2})(\\d{4})\\..+");
    
    for (const auto& entry : fs::directory_iterator(directory)) {
        std::string filename = entry.path().filename().string();
        std::smatch match;
        
        if (std::regex_match(filename, match, pattern)) {
            std::string xxx = match[1].str();
            std::string yyy = match[2].str();
            
            filesDict[yyy].push_back(std::make_pair(std::stoi(xxx), filename));
        }
    }
    
    // Sort each list by XXX
    for (auto& [key, value] : filesDict) {
        std::sort(value.begin(), value.end(), 
                 [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Extract just the filenames
        std::vector<std::string> sortedFilenames;
        for (const auto& pair : value) {
            sortedFilenames.push_back(pair.second);
        }
        
        value = sortedFilenames;
    }
    
    return filesDict;
}

// Main function
int main(int argc, char* argv[]) {
    Parameters params;
    
    auto videoNames = listAndSortFiles(getResourcePath("./"));
    
    for (const auto& [x, files] : videoNames) {
        std::vector<std::chrono::system_clock::time_point> times;
        auto output = mainPart(files, "pumps" + x + ".csv", params, times);
        
        if (!output.empty()) {
            createSupercut(files, output, times, "pumps" + x + ".mp4");
        }
    }
    
    return 0;
}

// Main processing function
std::vector<Pump> mainPart(const std::vector<std::string>& filenames, const std::string& filename, 
                          const Parameters& params, std::vector<std::chrono::system_clock::time_point>& timevec) {
    std::cout << "\nProcessing video " << filenames[0].substr(4, 4) << std::endl;
    
    char response;
    std::cout << "Do you want to pick an acceleration threshold value? (y/n): ";
    std::cin >> response;
    bool showPlots = (response == 'y' || response == 'Y');
    
    bool plots = showPlots || params.plots;
    
    // Extract GPMF streams
    std::vector<std::vector<uint8_t>> streams;
    for (const auto& x : filenames) {
        // This would be implemented using the GPMF parser library
        std::vector<uint8_t> stream; // Placeholder
        streams.push_back(stream);
    }
    
    // Combine streams
    std::vector<uint8_t> stream;
    for (const auto& s : streams) {
        stream.insert(stream.end(), s.begin(), s.end());
    }
    
    // Extract GPS and accelerometer data
    auto gpsBlocks = extractBlocks(stream, "GPS5");
    auto accelBlocks = extractBlocks(stream, "ACCL");
    
    // Extract device info and UTC times
    std::string devName = "GoPro"; // Placeholder
    std::string devID = "12345"; // Placeholder
    std::vector<double> utcTimes; // Placeholder
    
    std::cout << devName << " " << devID << std::endl;
    
    // Parse blocks into usable format
    std::vector<Eigen::MatrixXd> gpsData;
    std::vector<Eigen::MatrixXd> accelData;
    
    // Find index of first accurate timestamp
    int indexOfFirstAccurateTimestamp = 0;
    
    // Extract velocity data
    std::vector<Eigen::VectorXd> velocity;
    Eigen::VectorXd velocityConcat; // Concatenated velocity data
    
    // Process accelerometer data
    Eigen::MatrixXd accelDataConcat; // Concatenated accel data
    int accelN = accelDataConcat.rows();
    
    Eigen::VectorXd absAccel = Eigen::VectorXd::Zero(accelN);
    
    // Calculate absolute acceleration
    for (int i = 0; i < accelN; i++) {
        absAccel(i) = sqrt(pow(accelDataConcat(i, 0), 2) + 
                           pow(accelDataConcat(i, 1), 2) + 
                           pow(accelDataConcat(i, 2), 2));
    }
    
    // Interpolate timestamps
    std::vector<double> accelTimestamps;
    Eigen::VectorXd velocityT;
    Eigen::VectorXd gpsDataInterp;
    
    // Convert to datetime objects
    std::vector<std::chrono::system_clock::time_point> timeVec;
    
    // Apply cutoff filter
    Eigen::VectorXd absAccelCut = Eigen::VectorXd::Zero(absAccel.size());
    for (int i = 0; i < absAccel.size(); i++) {
        absAccelCut(i) = (absAccel(i) < params.cutOff) ? absAccel(i) : 0;
    }
    
    // Apply lowpass filter
    auto [b, a] = butter(4, params.passBand[1], "low", params.fs);
    Eigen::VectorXd lowpassedAccel = filtfilt(b, a, absAccelCut);
    lowpassedAccel.array() -= lowpassedAccel.mean();
    
    // Kernel definitions
    int kernelN = params.kernel_n;
    Eigen::VectorXd h = Eigen::VectorXd::Ones(kernelN) / kernelN;
    
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(kernelN, -M_PI, M_PI);
    
    // Create special kernel h3
    Eigen::VectorXd h3 = Eigen::VectorXd::Zero(kernelN);
    int thirdSize = kernelN / 3;
    
    for (int i = 0; i < thirdSize; i++) {
        h3(i) = 1.0 / kernelN;
    }
    for (int i = thirdSize; i < 2 * thirdSize; i++) {
        h3(i) = 0.1 / kernelN;
    }
    for (int i = 2 * thirdSize; i < kernelN; i++) {
        h3(i) = 1.0 / kernelN;
    }
    
    // Plot if needed
    if (plots) {
        // In a real implementation, this would create plots
        // using a C++ plotting library like matplotlib-cpp or similar
    }
    
    // Apply convolution
    Eigen::VectorXd unHilbAvgConv = plotNormedConvolution(lowpassedAccel.array().abs(), h, timeVec, "First Pass", plots);
    
    // Get pump threshold
    double pumpThresh = params.pumpThresh;
    if (plots) {
        std::cout << "Please enter a pump threshold value (graph must be closed first): ";
        std::cin >> pumpThresh;
    }
    
    // Extract pumps
    std::vector<Pump> pumps = extractPumps(unHilbAvgConv, pumpThresh);
    refinePumps(lowpassedAccel, pumps);
    addPumpMetaData(pumps, timeVec, indexOfFirstAccurateTimestamp);
    
    if (plots) {
        // Plot pump bounds
        // In a real implementation, this would create plots
    }
    
    if (pumps.empty()) {
        std::cout << "No pumps found in video " << filenames[0].substr(4, 4) << std::endl;
        return {};
    }
    
    // Create output dataframe (using CSV writing in C++)
    std::string videoName = filenames[0].substr(0, filenames[0].find_last_of('.'));
    std::string videoDir = fs::current_path().string() + "/" + videoName.substr(0, 2) + videoName.substr(4);
    
    if (!fs::exists(videoDir)) {
        fs::create_directories(videoDir);
    }
    
    // Write CSV file
    std::ofstream outputFile(videoDir + "/" + filename);
    outputFile << "start,end,videostartstamp,videoendstamp,duration\n";
    
    for (const auto& pump : pumps) {
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(pump.end - pump.start).count();
        
        // Format datetime to string
        auto start_time_t = std::chrono::system_clock::to_time_t(pump.start);
        auto end_time_t = std::chrono::system_clock::to_time_t(pump.end);
        
        std::stringstream start_ss, end_ss;
        start_ss << std::put_time(std::localtime(&start_time_t), "%Y-%m-%d %H:%M:%S");
        end_ss << std::put_time(std::localtime(&end_time_t), "%Y-%m-%d %H:%M:%S");
        
        outputFile << start_ss.str() << ","
                  << end_ss.str() << ","
                  << pump.start_str << ","
                  << pump.end_str << ","
                  << duration << "\n";
    }
    
    outputFile.close();
    
    return pumps;
}

// Extract pumps from data
std::vector<Pump> extractPumps(const Eigen::VectorXd& data, double pumpThresh) {
    std::vector<Pump> pumps;
    bool pumpFlag = false;
    int n = data.size();
    
    for (int i = 0; i < n - 1; i++) {
        if (pumpFlag) {
            if (data(i) < pumpThresh && data(i) > data(i+1)) {
                pumpFlag = false;
                pumps.back().endInd = i;
            }
        } else {
            if (data(i) > pumpThresh && data(i) < data(i+1)) {
                pumpFlag = true;
                Pump pump;
                pump.startInd = i;
                pump.endInd = n;
                pumps.push_back(pump);
            }
        }
    }
    
    // Calculate derivative
    Eigen::VectorXd deriv = Eigen::VectorXd::Zero(n - 1);
    for (int i = 0; i < n - 1; i++) {
        deriv(i) = data(i + 1) - data(i);
    }
    
    // Refine start and end indices
    for (auto& pump : pumps) {
        int x = pump.startInd;
        while (x > 0 && deriv(x) > 0) {
            x--;
        }
        pump.startInd = x;
        
        int x2 = pump.endInd;
        while (x2 < deriv.size() && deriv(x2) < 0) {
            x2++;
        }
        pump.endInd = x2;
    }
    
    return pumps;
}

// Refine pumps
void refinePumps(const Eigen::VectorXd& data, std::vector<Pump>& pumpList) {
    std::vector<int> removeList;
    
    for (int i = 0; i < pumpList.size(); i++) {
        int newStart = (pumpList[i].startInd + pumpList[i].endInd) / 2;
        int newEnd = newStart;
        
        while (newStart >= 0 && data(newStart) > 0) {
            newStart--;
        }
        while (newEnd < data.size() && data(newEnd) > 0) {
            newEnd++;
        }
        
        pumpList[i].startInd = newStart;
        pumpList[i].endInd = newEnd;
        
        if (newEnd - newStart < 402) {
            removeList.push_back(i);
        }
    }
    
    // Remove short pumps (in reverse to maintain indices)
    for (auto it = removeList.rbegin(); it != removeList.rend(); ++it) {
        pumpList.erase(pumpList.begin() + *it);
    }
}

// Add metadata to pumps
void addPumpMetaData(std::vector<Pump>& pumps, const std::vector<std::chrono::system_clock::time_point>& timevec, int firstTimeIndex) {
    auto firstTime = timevec[0] - std::chrono::seconds(firstTimeIndex);
    
    for (auto& pump : pumps) {
        pump.end = timevec[pump.endInd];
        pump.start = timevec[pump.startInd];
        pump.end_str = vidTimeStamp(firstTime, timevec[pump.endInd]);
        pump.start_str = vidTimeStamp(firstTime, timevec[pump.startInd]);
    }
}

// Convert time difference to video timestamp
std::string vidTimeStamp(const std::chrono::system_clock::time_point& datetime1, const std::chrono::system_clock::time_point& datetime2) {
    // Calculate the time difference
    auto timeDifference = datetime2 - datetime1;
    
    // Extract total seconds
    int totalSeconds = std::chrono::duration_cast<std::chrono::seconds>(timeDifference).count();
    
    // Convert to hours, minutes and seconds
    int hours = totalSeconds / 3600;
    int minutes = (totalSeconds % 3600) / 60;
    int seconds = totalSeconds % 60;
    
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(2) << hours << ":"
       << std::setfill('0') << std::setw(2) << minutes << ":"
       << std::setfill('0') << std::setw(2) << seconds;
    
    return ss.str();
}

// Convert HH:MM:SS to seconds
int hhmmssToSeconds(const std::string& hhmmss) {
    std::istringstream ss(hhmmss);
    std::string token;
    std::vector<int> times;
    
    while (std::getline(ss, token, ':')) {
        times.push_back(std::stoi(token));
    }
    
    return times[0] * 3600 + times[1] * 60 + times[2];
}

// Convolution with normalization
Eigen::VectorXd plotNormedConvolution(const Eigen::VectorXd& x, const Eigen::VectorXd& kernel, 
                                     const std::vector<std::chrono::system_clock::time_point>& timevec, 
                                     const std::string& label, bool plots) {
    Eigen::VectorXd output = convolve(x, kernel, "same");
    output.array() -= output.mean();
    
    if (plots) {
        // In a real implementation, this would create plots
    }
    
    return output;
}

// Create supercut of videos
void createSupercut(const std::vector<std::string>& videoUrls, const std::vector<Pump>& timestampPairs, 
                   const std::vector<std::chrono::system_clock::time_point>& videoTimestampVector, 
                   const std::string& outputFile, int timeOffset) {
    int totalOffset = 0;  // Keeps track of cumulative video duration
    
    char response;
    std::cout << "Do you want to add timestamps to the video? (y/n): ";
    std::cin >> response;
    bool addTimestamp = (response == 'y' || response == 'Y');
    
    std::cout << "Do you want to add 10s to beginning and end of the videos? (y/n): ";
    std::cin >> response;
    bool add10s = (response == 'y' || response == 'Y');
    
    try {
        for (int i = 0; i < videoUrls.size(); i++) {
            const auto& videoUrl = videoUrls[i];
            
            // Get video duration using FFmpeg
            AVFormatContext* formatContext = nullptr;
            if (avformat_open_input(&formatContext, videoUrl.c_str(), nullptr, nullptr) < 0) {
                std::cerr << "Could not open video file: " << videoUrl << std::endl;
                continue;
            }
            
            if (avformat_find_stream_info(formatContext, nullptr) < 0) {
                std::cerr << "Could not find stream info" << std::endl;
                avformat_close_input(&formatContext);
                continue;
            }
            
            double duration = formatContext->duration / (double)AV_TIME_BASE;
            avformat_close_input(&formatContext);
            
            // Create a directory for the video name if it doesn't exist
            std::string videoName = videoUrl.substr(0, videoUrl.find_last_of('.'));
            std::string videoDir = fs::current_path().string() + "/" + videoName.substr(0, 2) + videoName.substr(4);
            
            if (!fs::exists(videoDir)) {
                fs::create_directories(videoDir);
            }
            
            for (int j = 0; j < timestampPairs.size(); j++) {
                const auto& timestamps = timestampPairs[j];
                
                int startTime = hhmmssToSeconds(timestamps.start_str) - totalOffset + timeOffset;
                int endTime = std::min(hhmmssToSeconds(timestamps.end_str) - totalOffset + timeOffset, (int)duration);
                
                if (startTime < 0 || startTime > duration || endTime < 0) {
                    continue;
                }
                
                if (add10s) {
                    if (startTime > 10) {
                        startTime = startTime - 10;
                    } else {
                        startTime = 0;
                    }
                    
                    endTime = endTime + 10;
                }
                
                if (endTime > 0) {  // Only process if clip falls within this video
                    std::string tempFile = videoDir + "/" + videoName.substr(0, 2) + videoName.substr(4) + 
                                          "_" + std::to_string(j+1) + ".mp4";
                    
                    // Construct FFmpeg command
                    std::ostringstream ffmpegCmd;
                    ffmpegCmd << "ffmpeg -i \"" << videoUrl << "\" -ss " << startTime << " -to " << endTime;
                    
                    if (addTimestamp) {
                        // Get UNIX epoch time for timestamp
                        auto startEpoch = std::chrono::system_clock::to_time_t(timestamps.start);
                        
                        ffmpegCmd << " -vf \"drawtext=text='%{pts\\:localtime\\:" << startEpoch 
                                 << "\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S}':x=10:y=H-th-10:fontsize=24:"
                                 << "fontcolor=white@1.0:box=1:boxcolor=black@0.5:boxborderw=5\"";
                    }
                    
                    ffmpegCmd << " -c:a copy \"" << tempFile << "\" -loglevel quiet -y";
                    
                    // Execute FFmpeg command
                    int result = system(ffmpegCmd.str().c_str());
                    if (result != 0) {
                        std::cerr << "Error executing FFmpeg command" << std::endl;
                    }
                }
            }
            
            totalOffset += duration;  // Add current video duration to offset
        }
    } catch (const std::exception& e) {
        std::cerr << "Error processing video: " << e.what() << std::endl;
    }
}