
#ifndef TYPES_H
#define TYPES_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <memory>
#include <string>
#include <iostream>

struct plot_data {
double* x;
double* y;
size_t size;
const char* title;
char plot_type;

plot_data() : x(nullptr), y(nullptr), size(0), title(nullptr), plot_type('l') {}
};

struct Vector3D {
double x, y, z;
Vector3D(double x = 0.0, double y = 0.0, double z = 0.0) : x(x), y(y), z(z) {}
};

struct FileEntry {
const char* filename;
int sequence;
};

class AccelData {
public:
Vector3D *data;
double *timestamps;
size_t count;

AccelData() : data(nullptr), timestamps(nullptr), count(0) {}
~AccelData() {
    if (data) {
        free(data);
        data = nullptr;
    }
    if (timestamps) {
        free(timestamps);
        timestamps = nullptr;
    }
}

size_t size() const { return count; }
void reserve(size_t capacity) {
    data = (Vector3D*)malloc(capacity * sizeof(Vector3D));
    timestamps = (double*)malloc(capacity * sizeof(double));
}
};

class GPSData {
public:
double* speed_2d;
double* timestamps;
size_t count;

GPSData() : speed_2d(nullptr), timestamps(nullptr), count(0) {}
~GPSData() {
    if (speed_2d) {
        free(speed_2d);
        speed_2d = nullptr;
    }
    if (timestamps) {
        free(timestamps);
        timestamps = nullptr;
    }
}

size_t size() const { return count; }
void reserve(size_t capacity) {
    speed_2d = (double*)malloc(capacity * sizeof(double));
    timestamps = (double*)malloc(capacity * sizeof(double));
}
};

struct Pump {
int start_ind;
int end_ind;
double start_time;
double end_time;
std::string start_str;  // HH:MM:SS format
std::string end_str;    // HH:MM:SS format
double duration;

Pump()
    : start_ind(0), end_ind(0), start_time(0.0), end_time(0.0),
    start_str(""), end_str(""), duration(0.0) {}

friend std::ostream& operator<<(std::ostream& os, const Pump& obj) {
    return os << "Pump{start_ind: " << obj.start_ind << ", end_ind: " << obj.end_ind << ", start_time: " << obj.start_time << ", end_time: " << obj.end_time << ", duration: " << obj.duration << ", start_str: " << obj.start_str << ", end_str: " << obj.end_str  << "}";
}

};

struct PumpDetectionResult {
std::vector<Pump> pumps;
std::vector<double> signal; // Signal data for visualization

PumpDetectionResult() = default;

// Move constructor
PumpDetectionResult(PumpDetectionResult&& other) noexcept 
    : pumps(std::move(other.pumps)), signal(std::move(other.signal)) {
    other.signal.clear();
}

// Move assignment operator
PumpDetectionResult& operator=(PumpDetectionResult&& other) noexcept {
    if (this != &other) {
        // Clean up existing resources
        pumps = std::move(other.pumps);
        signal = std::move(other.signal);
    }
    return *this;
}

// Delete copy constructor and copy assignment to prevent double-free
PumpDetectionResult(const PumpDetectionResult&) = delete;
PumpDetectionResult& operator=(const PumpDetectionResult&) = delete;

~PumpDetectionResult() {
    signal.clear();
}
};


class FileGroup {
public:
    std::vector<std::string> filenames;
    std::string group_id;
    
    size_t size() const { return filenames.size(); }
    void addFile(const std::string& filename) { filenames.push_back(filename); }
};

struct 
Parameters {
    double accel_cutoff = 0.0;
    double lowpass_cutoff = 0.0;
    int fs = 0;
    int kernel_n_sec = 0;
    double pump_thresh = 0.0;
    int fine_kernel_n_sec = 0;
    bool plots = false;
    double fine_pump_thresh = 0.0;
    double velocity_thresh = 0.0;
    bool add_timestamps = false;
    bool add_10s = false;
    std::string video_quality = "medium";
    int vidQualityNum = 1; // Index for video quality selection in GUI
    std::string output_dir = "Windsurfing_Pump_Videos"; // Default output directory
    std::string input_dir = "."; // Default input directory
};

#endif // TYPES_H
