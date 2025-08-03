#ifndef GOPRO_PUMPING_PARSER_H
#define GOPRO_PUMPING_PARSER_H

// #include <cstdio>
// #include <cstdlib>
// #include <cstring>
// #include <cmath>
// #include <ctime>
// #include <vector>
// #include <memory>
#include "types.h"

#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <chrono>
#include <vector>

// #include <ostream>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <direct.h>
#define dirent _finddata_t
#define opendir(x) _findfirst(x, &find_data)
#define readdir(x) (_findnext(x, &find_data) == 0 ? &find_data : NULL)
#define closedir(x) _findclose(x)
#else
#include <dirent.h>
#include <sys/stat.h>
#include <regex.h>
#endif

// // For GPMF parsing
// extern "C" {
// #include "gpmf-parser/GPMF_parser.h"
// #include "gpmf-parser/GPMF_utils.h"
// #include "gpmf-parser/demo/GPMF_mp4reader.h"
// #include "libdspl-2.0/include/dspl.h"
// }


// Utility functions

std::string getResourcePath(const std::string& relative_path);
void trimWhitespace(std::string& str);
bool isGoProFile(const std::string& filename);

class ParamsWindow;

class MenuWindow;

class AppState {
public:
    const char* filenames = nullptr;
    Parameters params;
    bool show_params_window = false;
    bool show_menu_window = false;
    ParamsWindow* params_window;
    MenuWindow* menu_window;
    std::vector<FileGroup> file_groups;
    std::vector<PumpDetectionResult> pump_results; // Store results for each video group

    // Threading support
    std::mutex results_mutex;
    std::mutex cleanup_mutex;
    std::mutex futures_mutex;  // Dedicated mutex for futures container
    std::atomic<int> processed_groups{0};
    std::atomic<int> total_groups{0};
    std::atomic<bool> processing_active{false};
    std::atomic<bool> cleanup_done{false};
    std::atomic<bool> cleanup_thread_started{false};
    std::vector<std::future<bool>> processing_futures;
    std::thread rendering_thread;
    std::atomic<bool> should_exit{false};
    std::mutex progress_mutex;

    std::atomic<int> exported_videos{0}; // Track exported videos
    std::atomic<bool> export_active{false}; // Track if export is in progress
    std::vector<std::future<bool>> export_futures;
    std::mutex export_futures_mutex;
    std::atomic<bool> export_completed{false}; // Track if export is completed
    std::atomic<int> total_videos{0}; // Total number of videos processed


    AppState() : params_window(nullptr), menu_window(nullptr) {}
    
    ~AppState();

    void saveConfig();

    void resetToDefaults();
};

void start_processing_threads(AppState* app_state);


#endif // GOPRO_PUMPING_PARSER_H
