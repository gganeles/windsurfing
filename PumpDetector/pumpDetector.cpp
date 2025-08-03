#include "types.h"
#include "pumpDetector.h"
#include <iostream>
#include <vector>

extern "C" {
#include "gpmf-parser/GPMF_parser.h"
#include "gpmf-parser/GPMF_utils.h"
#include "gpmf-parser/GPMF_common.h"
#include "gpmf-parser/demo/GPMF_mp4reader.h"
#include "libdspl-2.0/include/dspl.h"
}

using namespace std;

struct MemoryPool {
    double* buffer;
    size_t total_size;
    size_t offset;
    
    MemoryPool(size_t size) : total_size(size), offset(0) {
        buffer = (double*)malloc(size * sizeof(double));
    }
    
    ~MemoryPool() {
        if (buffer) free(buffer);
    }
    
    double* allocate(size_t count) {
        if (offset + count > total_size) return nullptr;
        double* ptr = buffer + offset;
        offset += count;
        return ptr;
    }
    
    void reset() { offset = 0; }
};

void fast_abs_inplace(double* data, int size) {
    // Process in chunks of 4 for better cache utilization
    int i = 0;
    for (; i < size - 3; i += 4) {
        data[i] = fabs(data[i]);
        data[i+1] = fabs(data[i+1]);
        data[i+2] = fabs(data[i+2]);
        data[i+3] = fabs(data[i+3]);
    }
    // Handle remaining elements
    for (; i < size; i++) {
        data[i] = fabs(data[i]);
    }
}

int apply_butterworth_lowpass(double* input, int size, double cutoff, int fs, double** output) {
    *output = NULL;

    int res = RES_OK;
    double rp = 1.0; // Passband ripple in dB
    int ord = 4;       // Filter order
    cutoff = cutoff*.9;
    // Declare all pointers and scalars early to avoid C++ goto issues
    double *b = NULL, *a = NULL;
    double *bd = NULL, *ad = NULL;
    double *temp = NULL, *reversed = NULL;
    double a0 = 1.0;
    double warped = tan(M_PI * cutoff / fs);


    // Allocate memory for analog prototype coefficients
    b = (double*)malloc((ord + 1) * sizeof(double));
    a = (double*)malloc((ord + 1) * sizeof(double));
    if (!b || !a) goto cleanup_fail;

    // Design normalized analog lowpass prototype
    res = butter_ap(rp, ord, b, a);
    if (res != RES_OK) goto cleanup_fail;

    // Prewarp digital cutoff frequency to analog (rad/s)

    // Scale analog prototype to desired cutoff
    for (int i = 0; i <= ord; i++) {
        b[i] *= pow(warped, ord - i);
        a[i] *= pow(warped, ord - i);
    }

    // Allocate memory for digital coefficients
    bd = (double*)malloc((ord + 1) * sizeof(double));
    ad = (double*)malloc((ord + 1) * sizeof(double));
    if (!bd || !ad) goto cleanup_fail;

    // Convert to digital using bilinear transform
    res = bilinear(b, a, ord, bd, ad);
    if (res != RES_OK) goto cleanup_fail;

    // Normalize digital coefficients (make a[0] = 1)
    a0 = ad[0];
    for (int i = 0; i <= ord; i++) {
        bd[i] /= a0;
        ad[i] /= a0;
    }

    // Allocate buffers for filtering
    temp = (double*)malloc(size * sizeof(double));
    reversed = (double*)malloc(size * sizeof(double));
    if (!temp || !reversed) goto cleanup_fail;

    *output = (double*)malloc(size * sizeof(double));
    if (!*output) goto cleanup_fail;

    // Forward IIR filtering
    res = filter_iir(bd, ad, ord, input, size, temp);
    if (res != RES_OK) goto cleanup_fail;

    // Reverse the filtered signal
    for (int i = 0; i < size; i++) {
        reversed[i] = temp[size - 1 - i];
    }

    // Backward IIR filtering
    res = filter_iir(bd, ad, ord, reversed, size, *output);
    if (res != RES_OK) goto cleanup_fail;

    // Reverse again to restore original order
    for (int i = 0; i < size / 2; i++) {
        double t = (*output)[i];
        (*output)[i] = (*output)[size - 1 - i];
        (*output)[size - 1 - i] = t;
    }

    // Clean up and return
    free(b); free(a); free(bd); free(ad); free(temp); free(reversed);
    return RES_OK;

cleanup_fail:
    if (b) free(b);
    if (a) free(a);
    if (bd) free(bd);
    if (ad) free(ad);
    if (temp) free(temp);
    if (reversed) free(reversed);
    if (*output) { free(*output); *output = NULL; }
    return DSPL_ERROR_PTR;
}

int apply_convolution_dspl(double* signal, int signal_len, double* kernel, int kernel_len, double** output) {
    int conv_size = signal_len + kernel_len - 1;

    double* conv_result = (double*)malloc(conv_size * sizeof(double));
    if (!conv_result) return DSPL_ERROR_PTR;

    int res = conv(signal, signal_len, kernel, kernel_len, conv_result);
    if (res != RES_OK) {
        std::free(conv_result);
        return res;
    }

    // Allocate output buffer of original signal size
    *output = (double*)malloc(signal_len * sizeof(double));
    if (!*output) {
        std::free(conv_result);
        return DSPL_ERROR_PTR;
    }

    // Extract 'same' segment centered on the signal
    int start_idx = kernel_len / 2;
    for (int i = 0; i < signal_len; i++) {
        (*output)[i] = conv_result[start_idx + i];
    }

    std::free(conv_result);
    return RES_OK;
}

double* calculate_magnitude_vector(Vector3D* data, int size) {
    double* magnitude = (double*)malloc(size * sizeof(double));
    if (!magnitude) return NULL;
    
    for (int i = 0; i < size; i++) {
        // Use libdspl vector operations if available, otherwise manual calculation
        magnitude[i] = sqrt(data[i].x * data[i].x + 
                           data[i].y * data[i].y + 
                           data[i].z * data[i].z);
    }
    
    return magnitude;
}

// Fast in-place mean removal with vectorized operations
void fast_remove_mean_inplace(double* data, int size) {
    if (size == 0) return;
    
    // Calculate mean using Kahan summation for better numerical stability
    double mean = 0.0;
    double c = 0.0; // compensation for lost low-order bits
    
    for (int i = 0; i < size; i++) {
        double y = data[i] - c;
        double t = mean + y;
        c = (t - mean) - y;
        mean = t;
    }
    mean /= size;
    
    // Remove mean in chunks for better cache performance
    int i = 0;
    for (; i < size - 3; i += 4) {
        data[i] -= mean;
        data[i+1] -= mean;
        data[i+2] -= mean;
        data[i+3] -= mean;
    }
    for (; i < size; i++) {
        data[i] -= mean;
    }
}

// Optimized magnitude calculation with better cache usage
double* fast_calculate_magnitude(Vector3D* data, int size) {
    double* magnitude = (double*)malloc(size * sizeof(double));
    if (!magnitude) return nullptr;
    
    // Process in chunks for better cache locality
    for (int i = 0; i < size; i++) {
        double x = data[i].x;
        double y = data[i].y;
        double z = data[i].z;
        magnitude[i] = sqrt(x*x + y*y + z*z);
    }
    
    return magnitude;
}


double linear_interp(double x0, double x1, double y0, double y1, double x) {
    if (x1 == x0) return y0;  // Avoid division by zero
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0);
}

// Fast binary search-based scalar interpolation
double interp_scalar_fast(const double *x, const double *y, int n, double xq) {
    if (xq <= x[0]) return y[0];
    if (xq >= x[n - 1]) return y[n - 1];

    // Binary search for the correct interval
    int left = 0, right = n - 1;
    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (x[mid] <= xq) {
            left = mid;
        } else {
            right = mid;
        }
    }
    
    // Linear interpolation between x[left] and x[right]
    return linear_interp(x[left], x[right], y[left], y[right], xq);
}

// Legacy scalar interpolation (kept for compatibility)
double interp_scalar(const double *x, const double *y, int n, double xq) {
    return interp_scalar_fast(x, y, n, xq);
}

// Highly optimized vector interpolation
void interp_vector(const double *x, const double *y, int n, const double *xq, double *yq, int nq) {
    if (n <= 0 || nq <= 0) return;
    
    // Handle edge cases quickly
    if (n == 1) {
        for (int i = 0; i < nq; ++i) {
            yq[i] = y[0];
        }
        return;
    }
    
    // Check if query points are sorted (common case)
    bool sorted_queries = true;
    for (int i = 1; i < nq && sorted_queries; ++i) {
        if (xq[i] < xq[i-1]) sorted_queries = false;
    }
    
    if (sorted_queries) {
        // Optimized path for sorted queries - maintain search position
        int search_start = 0;
        
        for (int i = 0; i < nq; ++i) {
            double xq_val = xq[i];
            
            // Handle boundary cases
            if (xq_val <= x[0]) {
                yq[i] = y[0];
                continue;
            }
            if (xq_val >= x[n - 1]) {
                yq[i] = y[n - 1];
                continue;
            }
            
            // Start search from last found position (leveraging sorted nature)
            int left = search_start;
            int right = n - 1;
            
            // Find the interval containing xq_val
            while (left < n - 1 && x[left + 1] < xq_val) {
                left++;
            }
            right = left + 1;
            
            // Clamp to valid range
            if (right >= n) {
                right = n - 1;
                left = n - 2;
            }
            
            // Update search start for next iteration
            search_start = left;
            
            // Linear interpolation
            if (x[right] == x[left]) {
                yq[i] = y[left];
            } else {
                yq[i] = y[left] + (y[right] - y[left]) * (xq_val - x[left]) / (x[right] - x[left]);
            }
        }
    } else {
        // Fallback to binary search for unsorted queries
        for (int i = 0; i < nq; ++i) {
            yq[i] = interp_scalar_fast(x, y, n, xq[i]);
        }
    }
}

double calculate_mean(double* data, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

std::vector<Pump> extract_pumps(double* data, int size, double pump_thresh) {
    std::vector<Pump> pumps;
    pumps.reserve(MAX_PUMPS);

    int pump_flag = 0;
    for (int i = 0; i < size - 1; i++) {
        if (pump_flag) {
            if (data[i] < pump_thresh && data[i] > data[i + 1]) {
                pump_flag = 0;
                if (!pumps.empty()) {
                    pumps.back().end_ind = i;
                }
            }
        } else {
            if (data[i] > pump_thresh && data[i] < data[i + 1]) {
                pump_flag = 1;
                Pump new_pump;
                new_pump.start_ind = i;
                new_pump.end_ind = size - 1;  // default
                pumps.push_back(new_pump);

                if (pumps.size() >= MAX_PUMPS) break;
            }
        }
    }

    // Derivative
    double* deriv = (double*)malloc((size - 1) * sizeof(double));
    if (!deriv) {
        std::cerr << "Failed to allocate derivative buffer" << std::endl;
        return pumps;
    }

    for (int i = 0; i < size - 1; i++) {
        deriv[i] = data[i + 1] - data[i];
    }

    for (auto& pump : pumps) {
        int x = pump.start_ind;
        while (x > 0 && x < size - 1 && deriv[x] > 0) {
            x--;
        }
        pump.start_ind = std::max(0, x);

        int x2 = pump.end_ind;
        while (x2 < size - 2 && deriv[x2] < 0) {
            x2++;
        }
        pump.end_ind = std::min(size - 1, x2);
    }

    std::free(deriv);
    return pumps;
}

std::vector<Pump> refine_pumps(double* data, std::vector<Pump>& pumps, double thresh,
                   double* velocity, double vel_thresh) {
    std::vector<Pump> refined;
    refined.reserve(pumps.size());

    for (const auto& pump : pumps) {
        int start_ind = pump.start_ind;
        int end_ind = pump.end_ind;
        // Adjust start index
        while (data[start_ind] < thresh && start_ind != end_ind) {
            start_ind++;
        }
        if (start_ind == end_ind) continue;
        
        // Adjust end index
        while (data[end_ind] < thresh && end_ind > start_ind) {
            end_ind--;
        }
        // Check velocity threshold
        double vel_sum = 0.0;
        int vel_count = 0;
        for (int j = start_ind; j <= end_ind; j++) {
            vel_sum += velocity[j];
            vel_count++;
        }
        
        if (vel_count > 0 && (vel_sum / vel_count) > vel_thresh) {
            refined.push_back(pump);
            refined.back().start_ind = start_ind;
            refined.back().end_ind = end_ind;
        }
    }
    
    return refined;
}

std::string secondsToHHMMSS(int total_seconds) {
    int hours = total_seconds / 3600;
    int minutes = (total_seconds % 3600) / 60;
    int seconds = total_seconds % 60;
    
    char buffer[16];
    std::sprintf(buffer, "%02d:%02d:%02d", hours, minutes, seconds);
    return std::string(buffer);
}

void add_pump_metadata(vector<Pump>& pumps, double* timevec) {
    double first_time = timevec[0];

    for (auto& pump : pumps) {
        pump.start_time = timevec[pump.start_ind];
        pump.end_time = timevec[pump.end_ind];
        pump.duration = pump.end_time - pump.start_time;

        // Convert to video timestamp format
        double start_offset = pump.start_time - first_time;
        double end_offset = pump.end_time - first_time;
        pump.start_str = secondsToHHMMSS((int)start_offset);
        pump.end_str = secondsToHHMMSS((int)end_offset);
    }
}

// GPMF Processing Functions
int extract_gpmf_from_mp4(const char* filename, uint8_t** buffer, size_t* buffer_size) {
    // Open MP4 file using GPMF reader
    size_t mp4_handle = OpenMP4Source((char*)filename, MOV_GPMF_TRAK_TYPE, MOV_GPMF_TRAK_SUBTYPE, 0);
    if (mp4_handle == 0) {
        std::cout << "Error: Cannot open MP4 file " << filename << std::endl;
        return -1;
    }

    // Get number of payloads
    uint32_t payloads = GetNumberPayloads(mp4_handle);
    if (payloads == 0) {
        std::cout << "No GPMF payloads found in " << filename << std::endl;
        CloseSource(mp4_handle);
        return -1;
    }

    std::cout << "Found " << payloads << " GPMF payloads in " << filename << std::endl;

    // Calculate total buffer size needed for all payloads
    size_t total_size = 0;
    for (uint32_t i = 0; i < payloads; i++) {
        uint32_t payload_size = GetPayloadSize(mp4_handle, i);
        total_size += payload_size;
    }

    if (total_size == 0) {
        std::cout << "No GPMF payload data found" << std::endl;
        CloseSource(mp4_handle);
        return -1;
    }

    // Allocate buffer for all payloads
    *buffer = (uint8_t*)malloc(total_size);
    if (!*buffer) {
        std::cout << "Failed to allocate buffer for GPMF data" << std::endl;
        CloseSource(mp4_handle);
        return -1;
    }

    // Extract and concatenate all payloads
    size_t offset = 0;
    size_t payloadres = 0;
    size_t payloadsize = 0;
    uint32_t* payload = NULL;
    for (uint32_t i = 0; i < payloads; i++) {
        uint32_t payload_size = GetPayloadSize(mp4_handle, i);
        if (payload_size == 0) continue;

        // Get payload resource handle

        payloadsize = GetPayloadSize(mp4_handle, i);
        payloadres = GetPayloadResource(mp4_handle, payloadres, payloadsize);
        payload = GetPayload(mp4_handle, payloadres, i);
        if (payload == NULL)
            goto cleanup0;
        

        // Copy payload to buffer
        memcpy(*buffer + offset, payload, payload_size);
        offset += payload_size;
    }

    *buffer_size = offset;
    #ifdef DEBUG
    std::cout << "Extracted " << *buffer_size << " bytes of GPMF data from " << payloads << " payloads" << std::endl;
    #endif
    cleanup0:
		if (payloadres) FreePayloadResource(mp4_handle, payloadres);
		CloseSource(mp4_handle);
    return 0;
}


PumpDetectionResult main_part(FileGroup* filenames, Parameters params) {
    if (!filenames || filenames->size() == 0) {
        std::cout << "No video files to process" << std::endl;
        return {};
    }
    
    clock_t total_start = clock();
    
    // Pre-allocate memory pool for temporary arrays (reduces malloc overhead)
    size_t max_data_points = MAX_DATA_POINTS * filenames->size();
    MemoryPool memory_pool(max_data_points * 12); // Increased from 8 to 12 arrays worth of space
    
    // Process all video files in the group
    AccelData* combined_accel = new AccelData();
    GPSData* combined_gps = new GPSData();
    
    if (!combined_accel || !combined_gps) {
        std::cout << "Failed to allocate memory for AccelData or GPSData" << std::endl;
        if (combined_accel) delete combined_accel;
        if (combined_gps) delete combined_gps;
        return {};
    }
    
    combined_accel->reserve(max_data_points);
    combined_gps->reserve(max_data_points);
    
    if (!combined_accel->data || !combined_accel->timestamps || !combined_gps->speed_2d) {
        std::cout << "Failed to allocate memory for data arrays" << std::endl;
        delete combined_accel;
        delete combined_gps;
        return {};
    }
    
    combined_accel->count = 0;
    combined_gps->count = 0;
    vector<double> combined_timestamps;
    combined_timestamps.reserve(max_data_points);

    double time_offset = 0.0;
    
    // OPTIMIZATION: Batch process all GPMF data extraction    
    for (size_t file_idx = 0; file_idx < filenames->size(); file_idx++) {
        uint8_t* gpmf_buffer = nullptr;
        size_t gpmf_size = 0;
        
        if (extract_gpmf_from_mp4(filenames->filenames[file_idx].c_str(), &gpmf_buffer, &gpmf_size) != 0) {
            std::cout << "Warning: Could not extract GPMF from " << filenames->filenames[file_idx] << std::endl;
            continue;
        }
        
        // OPTIMIZATION: Parse all data types from same buffer in one pass
        GPMF_stream metadata_stream, *ms = &metadata_stream;
        int ret = GPMF_Init(ms, (uint32_t*)gpmf_buffer, (uint32_t)gpmf_size);
        
        if (ret == GPMF_OK) {
            // Parse accelerometer data efficiently
            GPMF_ResetState(ms);
            while (GPMF_OK == GPMF_FindNext(ms, STR2FOURCC("ACCL"), GPMF_RECURSE_LEVELS)) {
                uint32_t samples = GPMF_Repeat(ms);
                uint32_t elements = GPMF_ElementsInStruct(ms);
                
                if (elements >= 3 && combined_accel->count + samples < max_data_points) {
                    double* data = (double*)malloc(samples * elements * sizeof(double));
                    if (data && GPMF_ScaledData(ms, data, samples * elements * sizeof(double), 0, samples, GPMF_TYPE_DOUBLE) == GPMF_OK) {
                        
                        // Batch copy for better performance
                        for (uint32_t i = 0; i < samples; i++) {
                            combined_accel->data[combined_accel->count].x = data[i * elements + 0];
                            combined_accel->data[combined_accel->count].y = data[i * elements + 1];
                            combined_accel->data[combined_accel->count].z = data[i * elements + 2];
                            combined_accel->timestamps[combined_accel->count] = time_offset + (double)i / params.fs;
                            combined_accel->count++;
                        }
                        time_offset += (double)samples / params.fs;
                    }
                    if (data) free(data);
                }
            }
            
            // Parse GPS data efficiently
            GPMF_ResetState(ms);
            while (GPMF_OK == GPMF_FindNext(ms, STR2FOURCC("GPS5"), GPMF_RECURSE_LEVELS)) {
                uint32_t samples = GPMF_Repeat(ms);
                uint32_t elements = GPMF_ElementsInStruct(ms);
                
                if (elements >= 4 && combined_gps->count + samples < max_data_points) {
                    double* gps_scaled = (double*)malloc(samples * elements * sizeof(double));
                    if (gps_scaled && GPMF_ScaledData(ms, gps_scaled, samples * elements * sizeof(double), 0, samples, GPMF_TYPE_DOUBLE) == GPMF_OK) {
                        
                        // Batch copy GPS speed data
                        for (uint32_t i = 0; i < samples; i++) {
                            combined_gps->speed_2d[combined_gps->count++] = gps_scaled[i * elements + 3];
                        }
                    }
                    if (gps_scaled) free(gps_scaled);
                }
            }
            
            // Parse timestamp data efficiently  
            GPMF_ResetState(ms);
            while (GPMF_OK == GPMF_FindNext(ms, STR2FOURCC("GPSU"), GPMF_RECURSE_LEVELS)) {
                char* gpsu_raw = (char*)GPMF_RawData(ms);
                if (gpsu_raw) {
                    struct tm tm;
                    int sss;
                    if (sscanf(gpsu_raw, "%2d%2d%2d%2d%2d%2d.%3d", &tm.tm_year, &tm.tm_mon, &tm.tm_mday, &tm.tm_hour, &tm.tm_min, &tm.tm_sec, &sss) == 7) {
                        tm.tm_year += 100;
                        tm.tm_mon -= 1;
                        tm.tm_isdst = 0;
                        time_t unix_time_sec = mktime(&tm);
                        combined_timestamps.push_back((double)unix_time_sec + (sss / 1000.0));
                    }
                }
            }
        }
        
        std::free(gpmf_buffer);
    }
    
    if (combined_accel->count == 0) {
        std::cout << "No accelerometer data found in video files" << std::endl;
        delete combined_accel;
        delete combined_gps;
        return {};
    }
    
    std::cout << "Found " << combined_accel->count << " accel, " << combined_gps->count 
              << " GPS, " << combined_timestamps.size() << " timestamp samples" << std::endl;

    // OPTIMIZATION: Use fast magnitude calculation
    int data_size = combined_accel->count;
    double* abs_accel = fast_calculate_magnitude(combined_accel->data, data_size);
    if (!abs_accel) {
        std::cout << "Error calculating acceleration magnitude" << std::endl;
        delete combined_accel;
        delete combined_gps;
        return {};
    }
    
    // RESTORED: Original timestamp processing logic (this was causing the bug)
    int index_of_first_accurate_timestamp = 0;
    // Find the first nonzero timestamp (simulate "accurate" timestamps)
    for (size_t i = 0; i < combined_timestamps.size() - 1; i++) {
        if (combined_timestamps[i+1] - combined_timestamps[i] > 100000) { // adjust threshold as needed
            index_of_first_accurate_timestamp = i+1;
            continue;
        }
    }
    
    // Extend timestamps at the beginning with synthetic times if needed
    double* utcTimes = (double*)malloc(combined_timestamps.size() * sizeof(double));
    double* timeDiff = (double*)malloc(combined_timestamps.size() * sizeof(double));
    if (!timeDiff || !utcTimes) {
        std::cout << "Failed to allocate memory for time arrays" << std::endl;
        if (timeDiff) std::free(timeDiff);
        if (utcTimes) std::free(utcTimes);
        std::free(abs_accel);
        delete combined_accel;
        delete combined_gps;
        return {};
    }
    
    for (size_t i = index_of_first_accurate_timestamp; i < combined_timestamps.size()-1; i++) {
        timeDiff[i] = combined_timestamps[i+1] - combined_timestamps[i];
    }
    
    double mean_time_diff = calculate_mean(timeDiff + index_of_first_accurate_timestamp, 
                                         combined_timestamps.size() - 1 - index_of_first_accurate_timestamp);

    if (index_of_first_accurate_timestamp > 0) {
        double first_accurate = combined_timestamps[index_of_first_accurate_timestamp];
        for (int i = 0; i < index_of_first_accurate_timestamp; i++) {
            utcTimes[i] = first_accurate - mean_time_diff*(index_of_first_accurate_timestamp - i);
        }
        for (size_t i = index_of_first_accurate_timestamp; i < combined_timestamps.size(); i++) {
            utcTimes[i] = combined_timestamps[i];
        }
    } else {
        // Copy combined_timestamps data to utcTimes
        for (size_t i = 0; i < combined_timestamps.size(); i++) {
            utcTimes[i] = combined_timestamps[i];
        }
    }
    
    // OPTIMIZATION: Use memory pool for interpolation arrays
    double* timevec = memory_pool.allocate(data_size);
    double* velocity_interp = memory_pool.allocate(data_size);
    double* t = memory_pool.allocate(data_size);
    double* t2 = memory_pool.allocate(data_size);
    double* gpst2 = memory_pool.allocate(data_size);
    
    if (!timevec || !velocity_interp || !t || !t2 || !gpst2) {
        std::cout << "Memory pool exhausted for interpolation arrays" << std::endl;
        std::free(abs_accel);
        std::free(utcTimes);
        std::free(timeDiff);
        delete combined_accel;
        delete combined_gps;
        return {};
    }
    
    // RESTORED: Original interpolation logic (critical for correct timestamps)
    double upsampleScale = combined_timestamps.size() / (double)data_size;
    for (int i = 0; i < data_size; i++) {
        t[i] = (double)i;
    }
    for (int i = 0; i < data_size; i++) {
        t2[i] = ((double)i)*upsampleScale;
    }
    double gpsUpsampleScale = combined_gps->count / (double)data_size;
    for (int i = 0; i < data_size; i++) {
        gpst2[i] = ((double)i) * gpsUpsampleScale;
    }
    
    // Use the optimized interp_vector function
    interp_vector(t, utcTimes, combined_timestamps.size(), 
                 t2, timevec, data_size);
    interp_vector(t, combined_gps->speed_2d, 
                 combined_gps->count, gpst2, velocity_interp, data_size);
    

    std::free(utcTimes);
    std::free(timeDiff);
    
    // Clean up combined data structures early
    delete combined_accel;
    delete combined_gps;

    // OPTIMIZATION: In-place signal processing to reduce memory allocations
    
    // Apply cutoff filter in-place
    for (int i = 0; i < data_size; i++) {
        if (abs_accel[i] >= params.accel_cutoff) {
            abs_accel[i] = 0.0;
        }
    }

    // Apply Butterworth lowpass filter
    double* filtered_accel;
    int filter_res = apply_butterworth_lowpass(abs_accel, data_size, params.lowpass_cutoff, params.fs, &filtered_accel);
    std::free(abs_accel);
    
    if (filter_res != RES_OK || !filtered_accel) {
        std::cout << "Error applying lowpass filter" << std::endl;
        return {};
    }

    // OPTIMIZATION: Fast in-place mean removal
    fast_remove_mean_inplace(filtered_accel, data_size);
    
    // OPTIMIZATION: Reuse memory for abs operation
    fast_abs_inplace(filtered_accel, data_size);
    
    // Apply first convolution (reuse filtered_accel buffer as abs_filtered)
    double* abs_filtered = filtered_accel; // Reuse memory
    
    int kernel_n = params.kernel_n_sec * params.fs;
    double* kernel = memory_pool.allocate(kernel_n);
    if (!kernel) {
        std::cout << "Memory pool exhausted for kernel" << std::endl;
        std::free(filtered_accel);
        return {};
    }
    
    double kernel_val = 1.0 / kernel_n;
    for (int i = 0; i < kernel_n; i++) {
        kernel[i] = kernel_val;
    }
    
    double* convolved = nullptr;
    int conv_res = apply_convolution_dspl(abs_filtered, data_size, kernel, kernel_n, &convolved);
    
    if (conv_res != RES_OK || !convolved) {
        std::cout << "Error applying convolution" << std::endl;
        std::free(filtered_accel);
        return {};
    }

    // Fast mean removal on convolved data
    fast_remove_mean_inplace(convolved, data_size);

    // Extract pumps
    vector<Pump> pumps = extract_pumps(convolved, data_size, params.pump_thresh);

    if (pumps.empty()) {
        std::cout << "No pumps found in video group" << std::endl;
        std::free(filtered_accel);
        
        PumpDetectionResult empty_result;
        empty_result.pumps = pumps;
        empty_result.signal = vector<double>(convolved, convolved + data_size);
        std::free(convolved);
        return empty_result;
    }

    // Fine-tune with smaller kernel
    int kernel_n_2 = params.fine_kernel_n_sec * params.fs;
    double* kernel_2 = memory_pool.allocate(kernel_n_2);
    if (!kernel_2) {
        std::cout << "Memory pool exhausted for fine kernel" << std::endl;
        std::free(filtered_accel);
        std::free(convolved);
        return {};
    }
    
    double kernel_2_val = 1.0 / kernel_n_2;
    for (int i = 0; i < kernel_n_2; i++) {
        kernel_2[i] = kernel_2_val;
    }
    
    double* fine_convolved = nullptr;
    int fine_conv_res = apply_convolution_dspl(abs_filtered, data_size, kernel_2, kernel_n_2, &fine_convolved);
    
    if (fine_conv_res != RES_OK || !fine_convolved) {
        std::cout << "Error applying fine convolution" << std::endl;
        std::free(filtered_accel);
        std::free(convolved);
        return {};
    }

    fast_remove_mean_inplace(fine_convolved, data_size);

    // Refine pumps
    std::vector<Pump> refined_pumps = refine_pumps(fine_convolved, pumps, 
                                       params.fine_pump_thresh, velocity_interp, 
                                       params.velocity_thresh);
    
    if (!refined_pumps.empty()) {
        add_pump_metadata(refined_pumps, timevec);
    }
    
    // Cleanup
    PumpDetectionResult result;
    result.pumps = refined_pumps;
    
    // Allocate memory for signal data before copying
    result.signal = vector<double>(convolved, convolved + data_size);

    std::free(filtered_accel);
    std::free(convolved);
    std::free(fine_convolved);

    clock_t total_end = clock();
    double total_time = ((double)(total_end - total_start)) / CLOCKS_PER_SEC;
    std::cout << "Total processing time: " << total_time << " seconds" << std::endl << std::endl;

    
    return result;
}