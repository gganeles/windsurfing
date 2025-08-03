#include "goProPumpingParser.h"
#include "types.h"
#include "pumpDetector.h"
#include <GLFW/glfw3.h>

#include "imgui/imgui.h"
#include "implot/implot.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
#include "imgui/imgui_internal.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <cstdio>
#include <numeric>
#include <iomanip>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <future>
#include <chrono>
#include <filesystem>

using namespace std;

// Legacy C compatibility - for functions that still need C-style structs
extern "C"
{
#include "gpmf-parser/GPMF_parser.h"
#include "gpmf-parser/GPMF_utils.h"
#include "gpmf-parser/GPMF_common.h"
#include "gpmf-parser/demo/GPMF_mp4reader.h"
#include "libdspl-2.0/include/dspl.h"
#include <tinyfiledialogs.h>
}


#ifdef _WIN32
#include <windows.h>
#include <shobjidl.h>   // IFileDialog
#include <shlwapi.h>

// Converts std::wstring → std::string (UTF-8)
std::string WStringToString(const std::wstring& wstr) {
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string str(len, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), -1, &str[0], len, nullptr, nullptr);
    str.pop_back(); // remove trailing null
    return str;
}

// Converts std::string → std::wstring (UTF-8)
std::wstring StringToWString(const std::string& str) {
    int len = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring wstr(len, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], len);
    wstr.pop_back(); // remove trailing null
    return wstr;
}

/**
 * Opens a folder selection dialog.
 * @param title - The title of the dialog window (UTF-8).
 * @param initialDir - The initial folder to open to (UTF-8, optional).
 * @param owner - Optional HWND owner window.
 * @return Selected folder as std::string (UTF-8), or empty if canceled.
 */
std::string OpenFolderDialog_stdstring(const std::string& title,
                                       const std::string& initialDir = "",
                                       HWND owner = nullptr)
{
    std::string result;
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

    if (SUCCEEDED(hr)) {
        IFileDialog* pFileDialog = nullptr;
        hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                              IID_PPV_ARGS(&pFileDialog));

        if (SUCCEEDED(hr)) {
            DWORD options;
            pFileDialog->GetOptions(&options);
            pFileDialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);

            // Set custom dialog title
            std::wstring titleW = StringToWString(title);
            pFileDialog->SetTitle(titleW.c_str());

            // Optional: set initial folder
            if (!initialDir.empty()) {
                std::wstring initialW = StringToWString(initialDir);
                IShellItem* folderItem = nullptr;
                if (SUCCEEDED(SHCreateItemFromParsingName(initialW.c_str(), nullptr, IID_PPV_ARGS(&folderItem)))) {
                    pFileDialog->SetFolder(folderItem);
                    folderItem->Release();
                }
            }

            // Show dialog
            if (SUCCEEDED(pFileDialog->Show(owner))) {
                IShellItem* pItem = nullptr;
                if (SUCCEEDED(pFileDialog->GetResult(&pItem))) {
                    PWSTR pszPath = nullptr;
                    if (SUCCEEDED(pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszPath))) {
                        result = WStringToString(pszPath);
                        CoTaskMemFree(pszPath);
                    }
                    pItem->Release();
                }
            }

            pFileDialog->Release();
        }

        CoUninitialize();
    }

    return result; // empty if user canceled
}

#endif // _WIN32


// Forward declarations
void create_supercut(FileGroup *video_urls, vector<Pump> &pumps, Parameters args, AppState *app_state);
void wait_for_processing_completion(AppState *app_state);



// Function to wait for all processing threads to complete
void wait_for_export_completion(AppState *app_state)
{
    if (!app_state)
    {
        std::cerr << "Error: app_state is null in wait_for_export_completion" << std::endl;
        return;
    }

    std::cout << "Waiting for all export threads to complete..." << std::endl;

    // Simple approach: just wait for the processed_groups counter to reach total_groups
    // This avoids any issues with future handling
    const auto timeout_start = std::chrono::steady_clock::now();
    const auto max_wait_time = std::chrono::seconds(120); // Increased timeout to 2 minutes

    while (app_state->exported_videos.load() < app_state->total_videos.load() && !app_state->should_exit)
    {
        // Check for timeout
        auto now = std::chrono::steady_clock::now();
        if (now - timeout_start > max_wait_time)
        {
            std::cerr << "Warning: Timeout waiting for export threads to complete after 2 minutes" << std::endl;
            std::cerr << "Processed: " << app_state->exported_videos.load() << "/" << app_state->total_videos.load() << std::endl;
            break;
        }

        // Small sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Print progress every few seconds
        static auto last_progress_time = std::chrono::steady_clock::now();
        if (now - last_progress_time > std::chrono::seconds(5))
        {
            std::cout << "Progress: " << app_state->exported_videos.load() << "/"
                      << app_state->total_videos.load() << " videos completed" << std::endl;
            last_progress_time = now;
        }
    }

    // Now safely wait for and clear the futures container
    // At this point all threads should have completed based on the counter
    std::cout << "Waiting for all futures to complete and clearing container..." << std::endl;

    // Use the dedicated futures mutex to ensure thread-safe access
    {
        std::lock_guard<std::mutex> lock(app_state->export_futures_mutex);
        try
        {
            // Explicitly wait for each future to complete before clearing
            for (size_t i = 0; i < app_state->export_futures.size(); ++i)
            {
                auto &future = app_state->export_futures[i];
                if (future.valid())
                {
                    try
                    {
                        std::cout << "Waiting for future " << (i + 1) << " to complete..." << std::endl;

                        // Wait with longer timeout for each future
                        auto wait_result = future.wait_for(std::chrono::seconds(45)); // Increased timeout
                        if (wait_result == std::future_status::ready)
                        {
                            bool result = future.get(); // get() automatically waits and retrieves the result
                            std::cout << "Future " << (i + 1) << " completed with result: " << result << std::endl;
                        }
                        else
                        {
                            std::cerr << "Warning: Future " << (i + 1) << " did not complete within timeout" << std::endl;
                            // Don't call get() on a future that timed out to avoid blocking
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "Exception while waiting for future " << (i + 1) << ": " << e.what() << std::endl;
                    }
                    catch (...)
                    {
                        std::cerr << "Unknown exception while waiting for future " << (i + 1) << std::endl;
                    }
                }
                else
                {
                    std::cout << "Future " << (i + 1) << " is already invalid" << std::endl;
                }
            }

            // Now it's safe to clear the container
            app_state->export_futures.clear();
            std::cout << "Futures container cleared successfully." << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception while clearing futures: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception while clearing futures" << std::endl;
        }
    }

    // Mark processing as inactive after all cleanup is complete
    app_state->export_active = false;
    app_state->export_completed=true;
    std::cout << "All export threads completed successfully." << std::endl;
}

void save_pumps_to_csv(FileGroup &file_group, const vector<Pump> &pumps, const Parameters &params) {
            // Create output directory for this group
    std::string filename = params.output_dir + "/GL" + file_group.group_id + "/pumps" + file_group.group_id + ".csv";
    FILE* file = fopen(filename.c_str(), "w");
    if (!file) {
        std::cout << "Error: Cannot create output file: " << filename << std::endl;
        return;
    }
    
    std::fprintf(file, "start,end,videostartstamp,videoendstamp,duration\n");
    
    for (size_t i = 0; i < pumps.size(); i++) {
        time_t start_time = (time_t)pumps[i].start_time;
        time_t end_time = (time_t)pumps[i].end_time;
        struct tm* start_tm = localtime(&start_time);
        struct tm* end_tm = localtime(&end_time);

        char start_str[32], end_str[32];
        strftime(start_str, sizeof(start_str), "%Y-%m-%d %H:%M:%S", start_tm);
        strftime(end_str, sizeof(end_str), "%Y-%m-%d %H:%M:%S", end_tm);
        
        std::fprintf(file, "%s,%s,%s,%s,%.2f\n", 
                start_str, end_str, pumps[i].start_str.c_str(), pumps[i].end_str.c_str(), pumps[i].duration);
    }
    
    fclose(file);
}


bool export_file_group_thread(AppState *app_state, FileGroup file_group, int group_index){
    if (!app_state)
    {
        std::cerr << "Error: app_state is null in process_file_group_thread" << std::endl;
        return false;
    }

    // Check if we should exit early
    if (app_state->should_exit)
    {
        std::cout << "Thread " << group_index << " exiting early due to shutdown request" << std::endl;
        return false;
    }

    
    try
    {
        std::cout << "Exporting csv of group " << file_group.group_id << " with "
                  << file_group.size() << " files" << std::endl;

        // Create output directory for this group
        char output_Directory[256];
        snprintf(output_Directory, sizeof(output_Directory), "%s\\GL%s", app_state->params.output_dir.c_str(), file_group.group_id.c_str());

        #ifdef _WIN32
                char mkdir_cmd[128];
                snprintf(mkdir_cmd, sizeof(mkdir_cmd), "if not exist \"%s\" mkdir \"%s\"",
                        output_Directory, output_Directory);
                system(mkdir_cmd);
        #else
                char mkdir_cmd[128];
                snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p \"%s\"", output_Directory);
                system(mkdir_cmd);
        #endif

        // Check again if we should exit before the heavy processing
        if (app_state->should_exit)
        {
            std::cout << "Thread " << group_index << " exiting before processing due to shutdown request" << std::endl;
            return false;
        }

        // Export the data from the file group's pumps
        save_pumps_to_csv(file_group, app_state->pump_results[group_index].pumps, app_state->params);

        // Create supercut if pumps were found
        if (app_state->pump_results[group_index].pumps.size() > 0) {
            create_supercut(&file_group, app_state->pump_results[group_index].pumps, app_state->params, app_state);
        }

        // Check if we should exit before storing results
        if (app_state->should_exit)
        {
            std::cout << "Thread " << group_index << " exiting before storing results due to shutdown request" << std::endl;
            return false;
        }

        std::cout << "Group " << (group_index + 1) << " export thread completed successfully" << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing group " << (group_index + 1) << ": " << e.what() << std::endl;
        return false;
    }
    catch (...)
    {
        std::cerr << "Unknown error processing group " << (group_index + 1) << std::endl;
        return false;
    }
}

// Function to start processing all file groups in separate threads
void start_exporting_threads(AppState *app_state)
{
    if (app_state->file_groups.empty())
    {
        std::cout << "No file groups to process" << std::endl;
        return;
    }

    // Ensure previous processing is completely finished before starting new one
    std::cout << "Ensuring previous processing is complete..." << std::endl;

    // Wait for any ongoing processing to complete
    if (app_state->export_active.load())
    {
        std::cout << "Waiting for export to complete..." << std::endl;
        wait_for_export_completion(app_state);
    }

    // Reset exit flag for new export run
    app_state->should_exit = false;

    // Reset export state
    app_state->export_active = true;
    app_state->exported_videos = 0;

    app_state->total_videos = 0;
    for (const auto &pumpResult : app_state->pump_results)
    {
        app_state->total_videos.fetch_add(pumpResult.pumps.size());
        if (pumpResult.pumps.size() > 0)
        {
            app_state->total_videos++; // Count the supercut as one video
        }
    }
    cout<<"Total videos to Export: "<<app_state->total_videos<<endl;
    app_state->cleanup_done = false; // Reset cleanup state


    // Clear any existing futures safely with longer timeout
    std::cout << "Clearing previous futures..." << std::endl;
    {
        std::lock_guard<std::mutex> lock(app_state->futures_mutex);
        // Wait for any existing futures to complete before clearing with longer timeout
        for (auto &future : app_state->processing_futures)
        {
            if (future.valid())
            {
                try
                {
                    auto wait_result = future.wait_for(std::chrono::seconds(10)); // Increased timeout
                    if (wait_result == std::future_status::ready)
                    {
                        future.get();
                    }
                    else
                    {
                        std::cerr << "Warning: Future did not complete within 10 seconds, forcing cleanup" << std::endl;
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Exception during future cleanup: " << e.what() << std::endl;
                }
                catch (...)
                {
                    std::cerr << "Unknown exception during future cleanup" << std::endl;
                }
            }
        }
        app_state->processing_futures.clear();
    }

    std::cout << "Starting processing of " << app_state->total_groups << " file groups" << std::endl;

    // Create a thread for each file group
    for (int i = 0; i < app_state->total_groups; i++)
    {
        int videos_in_group = 1+app_state->file_groups[i].size(); // +1 for the supercut
        if (app_state->should_exit.load())
        {
            std::cout << "Stopping thread creation due to exit request" << std::endl;
            break;
        }

        FileGroup file_group_copy = app_state->file_groups[i];

        try
        {
            std::future<bool> future = std::async(std::launch::async,
                                                  [app_state, file_group_copy, i]() mutable
                                                  {
                                                      return export_file_group_thread(app_state, file_group_copy, i);
                                                  });

            // Use the futures mutex when adding to the container
            {
                std::lock_guard<std::mutex> lock(app_state->export_futures_mutex);
                app_state->export_futures.push_back(std::move(future));
            }
        }
        catch (const std::exception &e)
        {
            app_state->exported_videos+= videos_in_group; // Increment exported count even if thread creation fails
            std::cerr << "Failed to create processing thread " << i << ": " << e.what() << std::endl;
        }
        catch (...)
        {
            app_state->exported_videos+= videos_in_group; // Increment exported count even if thread creation fails
            std::cerr << "Unknown error creating processing thread " << i << std::endl;
        }
    }
    std::cout << "All processing threads started successfully" << std::endl;
}

bool process_file_group_thread(AppState *app_state, FileGroup file_group, int group_index)
{
    if (!app_state)
    {
        std::cerr << "Error: app_state is null in process_file_group_thread" << std::endl;
        return false;
    }

    // Check if we should exit early
    if (app_state->should_exit)
    {
        std::cout << "Thread " << group_index << " exiting early due to shutdown request" << std::endl;
        return false;
    }

    try
    {
        std::cout << "Starting processing of group " << file_group.group_id << " with "
                  << file_group.size() << " files" << std::endl;

        // Create output filename for this group
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "GL%s/pumps%s.csv",
                 file_group.group_id.c_str(), file_group.group_id.c_str());



        // Check again if we should exit before the heavy processing
        if (app_state->should_exit)
        {
            std::cout << "Thread " << group_index << " exiting before processing due to shutdown request" << std::endl;
            return false;
        }

        // Process the file group
        PumpDetectionResult result = main_part(&file_group, app_state->params);

        // Check if we should exit before storing results
        if (app_state->should_exit)
        {
            std::cout << "Thread " << group_index << " exiting before storing results due to shutdown request" << std::endl;
            return false;
        }

        // Store result safely
        size_t pump_count = result.pumps.size(); // Store count before moving
        {
            std::lock_guard<std::mutex> lock(app_state->results_mutex);
            if (group_index >= (int)app_state->pump_results.size())
            {
                app_state->pump_results.resize(group_index + 1);
            }
            app_state->pump_results[group_index] = std::move(result);
        }

        std::cout << "Completed processing group " << (group_index + 1) << " - Found "
                  << pump_count << " pumps" << std::endl;

        // Create supercut if pumps were found and video cutting is enabled
        // if (result.pumps.size() > 0 && (app_state->params.add_timestamps || app_state->params.add_10s)) {
        //     std::cout << "Creating video cuts for group " << (group_index + 1) << std::endl;
        //     // Convert the result pumps to the format expected by create_supercut
        //     vector<Pump> pumps_vector(result.pumps.begin(), result.pumps.end());
        //     create_supercut(&file_group, pumps_vector, app_state->params);
        // }

        // Update progress AFTER all processing is complete
        ++app_state->processed_groups;

        std::cout << "Group " << (group_index + 1) << " processing thread completed successfully" << std::endl;
        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error processing group " << (group_index + 1) << ": " << e.what() << std::endl;
        ++app_state->processed_groups;
        return false;
    }
    catch (...)
    {
        std::cerr << "Unknown error processing group " << (group_index + 1) << std::endl;
        ++app_state->processed_groups;
        return false;
    }
}

// Function to start processing all file groups in separate threads
void start_processing_threads(AppState *app_state)
{
    if (app_state->file_groups.empty())
    {
        std::cout << "No file groups to process" << std::endl;
        return;
    }

    // Ensure previous processing is completely finished before starting new one
    std::cout << "Ensuring previous processing is complete..." << std::endl;

    // Wait for any ongoing processing to complete
    if (app_state->processing_active.load())
    {
        std::cout << "Waiting for previous processing to complete..." << std::endl;
        wait_for_processing_completion(app_state);
    }

    // Reset exit flag for new processing run
    app_state->should_exit = false;

    // Reset processing state
    app_state->processing_active = true;
    app_state->processed_groups = 0;
    app_state->cleanup_done = false;
    app_state->cleanup_thread_started = false;
    app_state->total_groups.store((int)(app_state->file_groups.size()));

    // Clear previous results safely
    std::cout << "Clearing previous results..." << std::endl;
    {
        std::lock_guard<std::mutex> lock(app_state->results_mutex);
        std::cout << "Lock entered" << std::endl;

        // Create a new empty vector and swap it to avoid destructor issues
        std::vector<PumpDetectionResult> empty_results;
        cout << "Creating empty results vector" << endl;
        empty_results.reserve(app_state->total_groups.load());
        cout << "reserved" << endl;
        app_state->pump_results.swap(empty_results);
    }

    // Clear any existing futures safely with longer timeout
    std::cout << "Clearing previous futures..." << std::endl;
    {
        std::lock_guard<std::mutex> lock(app_state->futures_mutex);
        // Wait for any existing futures to complete before clearing with longer timeout
        for (auto &future : app_state->processing_futures)
        {
            if (future.valid())
            {
                try
                {
                    auto wait_result = future.wait_for(std::chrono::seconds(10)); // Increased timeout
                    if (wait_result == std::future_status::ready)
                    {
                        future.get();
                    }
                    else
                    {
                        std::cerr << "Warning: Future did not complete within 10 seconds, forcing cleanup" << std::endl;
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Exception during future cleanup: " << e.what() << std::endl;
                }
                catch (...)
                {
                    std::cerr << "Unknown exception during future cleanup" << std::endl;
                }
            }
        }
        app_state->processing_futures.clear();
    }

    std::cout << "Starting processing of " << app_state->total_groups << " file groups" << std::endl;

    // Create a thread for each file group
    for (int i = 0; i < app_state->total_groups; i++)
    {
        if (app_state->should_exit)
        {
            std::cout << "Stopping thread creation due to exit request" << std::endl;
            break;
        }

        FileGroup file_group_copy = app_state->file_groups[i];

        try
        {
            std::future<bool> future = std::async(std::launch::async,
                                                  [app_state, file_group_copy, i]() mutable
                                                  {
                                                      return process_file_group_thread(app_state, file_group_copy, i);
                                                  });

            // Use the futures mutex when adding to the container
            {
                std::lock_guard<std::mutex> lock(app_state->futures_mutex);
                app_state->processing_futures.push_back(std::move(future));
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Failed to create processing thread " << i << ": " << e.what() << std::endl;
            ++app_state->processed_groups; // Count as processed to avoid hanging
        }
        catch (...)
        {
            std::cerr << "Unknown error creating processing thread " << i << std::endl;
            ++app_state->processed_groups; // Count as processed to avoid hanging
        }
    }
    std::cout << "All processing threads started successfully" << std::endl;
}

// Function to wait for all processing threads to complete
void wait_for_processing_completion(AppState *app_state)
{
    if (!app_state)
    {
        std::cerr << "Error: app_state is null in wait_for_processing_completion" << std::endl;
        return;
    }

    std::cout << "Waiting for all processing threads to complete..." << std::endl;

    // Simple approach: just wait for the processed_groups counter to reach total_groups
    // This avoids any issues with future handling
    const auto timeout_start = std::chrono::steady_clock::now();
    const auto max_wait_time = std::chrono::seconds(120); // Increased timeout to 2 minutes

    while (app_state->processed_groups.load() < app_state->total_groups.load() && !app_state->should_exit)
    {
        // Check for timeout
        auto now = std::chrono::steady_clock::now();
        if (now - timeout_start > max_wait_time)
        {
            std::cerr << "Warning: Timeout waiting for processing threads to complete after 2 minutes" << std::endl;
            std::cerr << "Processed: " << app_state->processed_groups.load() << "/" << app_state->total_groups.load() << std::endl;
            break;
        }

        // Small sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Print progress every few seconds
        static auto last_progress_time = std::chrono::steady_clock::now();
        if (now - last_progress_time > std::chrono::seconds(5))
        {
            std::cout << "Progress: " << app_state->processed_groups.load() << "/"
                      << app_state->total_groups.load() << " groups completed" << std::endl;
            last_progress_time = now;
        }
    }

    // Now safely wait for and clear the futures container
    // At this point all threads should have completed based on the counter
    std::cout << "Waiting for all futures to complete and clearing container..." << std::endl;

    // Use the dedicated futures mutex to ensure thread-safe access
    {
        std::lock_guard<std::mutex> lock(app_state->futures_mutex);
        try
        {
            // Explicitly wait for each future to complete before clearing
            for (size_t i = 0; i < app_state->processing_futures.size(); ++i)
            {
                auto &future = app_state->processing_futures[i];
                if (future.valid())
                {
                    try
                    {
                        std::cout << "Waiting for future " << (i + 1) << " to complete..." << std::endl;

                        // Wait with longer timeout for each future
                        auto wait_result = future.wait_for(std::chrono::seconds(45)); // Increased timeout
                        if (wait_result == std::future_status::ready)
                        {
                            bool result = future.get(); // get() automatically waits and retrieves the result
                            std::cout << "Future " << (i + 1) << " completed with result: " << result << std::endl;
                        }
                        else
                        {
                            std::cerr << "Warning: Future " << (i + 1) << " did not complete within timeout" << std::endl;
                            // Don't call get() on a future that timed out to avoid blocking
                        }
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "Exception while waiting for future " << (i + 1) << ": " << e.what() << std::endl;
                    }
                    catch (...)
                    {
                        std::cerr << "Unknown exception while waiting for future " << (i + 1) << std::endl;
                    }
                }
                else
                {
                    std::cout << "Future " << (i + 1) << " is already invalid" << std::endl;
                }
            }

            // Now it's safe to clear the container
            app_state->processing_futures.clear();
            std::cout << "Futures container cleared successfully." << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception while clearing futures: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception while clearing futures" << std::endl;
        }
    }

    // Mark processing as inactive after all cleanup is complete
    app_state->processing_active = false;
    std::cout << "All processing threads completed successfully." << std::endl;
}

int multiPlottingWindow(vector<plot_data> data, const char *title = "Plot Window")
{
    // Initialize GLFW and ImGui for plotting
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow *window = glfwCreateWindow(1280, 720, "ImPlot Example", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Your plotting code here
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

        ImGui::Begin(title);
        ImVec2 content_size = ImGui::GetContentRegionAvail();
        if (ImPlot::BeginPlot("##FullPlot", content_size))
        {
            for (const auto &plot : data)
            {
                if (plot.x && plot.y && plot.size > 0)
                {
                    if (plot.plot_type == 'l')
                    {
                        ImPlot::PlotLine(plot.title, plot.x, plot.y, plot.size);
                    }
                    else if (plot.plot_type == 's')
                    {
                        ImPlot::PlotScatter(plot.title, plot.x, plot.y, plot.size);
                    }
                }
            }
            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

int hhmmss_to_seconds(const std::string &hhmmss)
{
    std::istringstream stream(hhmmss);
    int h, m, s;
    char c1, c2;

    if (stream >> h >> c1 >> m >> c2 >> s &&
        c1 == ':' && c2 == ':' &&
        m >= 0 && m < 60 && s >= 0 && s < 60)
    {
        return h * 3600 + m * 60 + s;
    }

    std::cerr << "Failed to parse: " << hhmmss << "\n";
    return -1;
}

void trimWhitespace(std::string &str)
{
    // Remove trailing whitespace
    str.erase(std::find_if(str.rbegin(), str.rend(),
                           [](unsigned char ch)
                           { return !std::isspace(ch); })
                  .base(),
              str.end());

    // Remove leading whitespace
    str.erase(str.begin(),
              std::find_if(str.begin(), str.end(),
                           [](unsigned char ch)
                           { return !std::isspace(ch); }));
}

std::string getResourcePath(const std::string &relative_path)
{
    char buffer[MAX_PATH_LEN];
    getcwd(buffer, sizeof(buffer));
    return std::string(buffer) + "/" + relative_path;
}

bool isGoProFile(const std::string &filename)
{
    if (filename.length() < 10)
        return false;
    if (filename[0] != 'G')
        return false;

    // Check if positions 2-7 are digits
    for (int i = 2; i <= 7; ++i)
    {
        if (!isdigit(filename[i]))
            return false;
    }

    // Check valid extensions
    std::string ext = filename.substr(filename.length() - 4);
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == ".mp4" || ext == ".lrv");
}

void lowerAndTrim(std::string &str)
{
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    trimWhitespace(str);
};

Parameters parse_params(const char *filename)
{
    Parameters params = {0};

    FILE *file = fopen(filename, "r");
    if (!file)
    {
        std::cout << "Error: Cannot open params file: " << filename << std::endl;
        return params;
    }

    char line[256];
    while (fgets(line, sizeof(line), file))
    {
        char *colon = strchr(line, ':');
        if (colon)
        {
            *colon = '\0';
            std::string key = line;
            std::string value = colon + 1;

            trimWhitespace(key);
            trimWhitespace(value);

            if (key == "accelCutOff")
            {
                params.accel_cutoff = atof(value.c_str());
            }
            else if (key == "lowPassCutoff")
            {
                params.lowpass_cutoff = atof(value.c_str());
            }
            else if (key == "fs")
            {
                params.fs = atoi(value.c_str());
            }
            else if (key == "kernel_n (sec)")
            {
                params.kernel_n_sec = atoi(value.c_str());
            }
            else if (key == "pumpThresh")
            {
                params.pump_thresh = atof(value.c_str());
            }
            else if (key == "fine_kernel_n (sec)")
            {
                params.fine_kernel_n_sec = atoi(value.c_str());
            }
            else if (key == "plots")
            {
                // Convert "True" or "False" to 1 or 0
                lowerAndTrim(value);
                params.plots = (value == "true") ? 1 : 0;
            }
            else if (key == "finePumpThresh")
            {
                params.fine_pump_thresh = atof(value.c_str());
            }
            else if (key == "velocityThresh")
            {
                params.velocity_thresh = atof(value.c_str());
            }
            else if (key == "addTimestamps")
            {
                lowerAndTrim(value);
                params.add_timestamps = (value == "true") ? 1 : 0;
            }
            else if (key == "add10sec")
            {
                lowerAndTrim(value);
                params.add_10s = (value == "true") ? 1 : 0;
            }
            else if (key == "videoQuality")
            {
                lowerAndTrim(value);
                const char *video_quality_options[] = {"low", "medium", "high", "very high"};

                params.video_quality = value;

                params.vidQualityNum = -1;
                for (int n = 0; n < 4; ++n)
                {
                    if (params.video_quality == video_quality_options[n])
                    {
                        params.vidQualityNum = n;
                        break;
                    }
                }
            }
            else if (key == "outputDir")
            {
                trimWhitespace(value);
                if (value.back() != '\\')
                {
                    value += '\\'; // Ensure directory ends with '\'
                }
                params.output_dir = value;
            }
            else if (key == "inputDir")
            {
                trimWhitespace(value);
                params.input_dir = value;
            }
        }
    }

    fclose(file);
    return params;
}

// Video encoding parameter structure for different quality levels
struct VideoEncodingParams
{
    std::string preset;
    int crf;
    std::string additional_params;
    std::string codec;
};

/**
 * Get video encoding parameters based on quality level
 *
 * Quality levels:
 * - "low": Fast encoding, higher compression (CRF 28, superfast preset)
 *   Good for: Quick processing, smaller file sizes, preview videos
 *
 * - "medium": Balanced encoding speed and quality (CRF 23, fast preset) [DEFAULT]
 *   Good for: Most use cases, reasonable quality and processing time
 *
 * - "high": Better quality, slower encoding (CRF 20, medium preset)
 *   Good for: Final videos where quality is important
 *
 * - "very high": Best quality, slowest encoding (CRF 18, slow preset)
 *   Good for: Archive quality, maximum visual fidelity
 *
 * - "copy": Stream copy, no re-encoding (fastest)
 *   Good for: When no quality changes are needed
 *
 * @param quality_level String indicating desired quality level
 * @return VideoEncodingParams struct with appropriate FFmpeg parameters
 */
VideoEncodingParams get_video_encoding_params(const std::string &quality_level)
{
    VideoEncodingParams params;

    // Convert to lowercase for comparison
    std::string quality = quality_level;
    std::transform(quality.begin(), quality.end(), quality.begin(), ::tolower);

    if (quality == "low")
    {
        // Low quality: Ultra-fast encoding, maximum compression
        params.preset = "ultrafast";
        params.crf = 32;
        params.codec = "libx264";
        params.additional_params = "-tune zerolatency -threads 0 -x264-params \"me=dia:subme=0:ref=1:analyse=none:trellis=0:no-fast-pskip=1:8x8dct=0:aq-mode=0:bframes=0:cabac=0\"";
    }
    else if (quality == "medium")
    {
        // Medium quality: Balanced encoding speed and quality
        params.preset = "fast";
        params.crf = 23;
        params.codec = "libx264";
        params.additional_params = "-tune fastdecode -threads 0 -x264-params \"me=hex:subme=2:ref=2:analyse=i4x4:trellis=0:no-fast-pskip=0:8x8dct=0:aq-mode=1\"";
    }
    else if (quality == "high")
    {
        // High quality: Better quality, slower encoding
        params.preset = "medium";
        params.crf = 20;
        params.codec = "libx264";
        params.additional_params = "-tune film -threads 0 -x264-params \"me=umh:subme=6:ref=3:analyse=all:trellis=1:no-fast-pskip=0:8x8dct=1:aq-mode=1\"";
    }
    else if (quality == "very high")
    {
        // Very high quality: Best quality, slowest encoding
        params.preset = "slow";
        params.crf = 18;
        params.codec = "libx264";
        params.additional_params = "-tune film -threads 0 -x264-params \"me=umh:subme=8:ref=5:analyse=all:trellis=2:no-fast-pskip=0:8x8dct=1:aq-mode=1:psy-rd=1.0,0.0\"";
    }
    else
    {
        // Default to medium quality for unknown values
        params.preset = "fast";
        params.crf = 23;
        params.codec = "libx264";
        params.additional_params = "-tune fastdecode -threads 0 -x264-params \"me=hex:subme=2:ref=2:analyse=i4x4:trellis=0:no-fast-pskip=0:8x8dct=0:aq-mode=1\"";
    }

    return params;
}

// Execute FFmpeg command with error checking and progress indication
// Global flag to track if FFmpeg availability has been checked
static bool ffmpeg_checked = false;
static std::mutex ffmpeg_check_mutex;

int execute_ffmpeg_command(const char *cmd)
{
    // Check if ffmpeg is available (only once per program run)
    {
        std::lock_guard<std::mutex> lock(ffmpeg_check_mutex);
        if (!ffmpeg_checked)
        {
            int ffmpeg_check = system("ffmpeg -version >nul 2>&1");
            if (ffmpeg_check != 0)
            {
                std::cout << "Error: FFmpeg not found in PATH. Please install FFmpeg." << std::endl;
                std::cout << "Download from: https://ffmpeg.org/download.html" << std::endl;
                return -1;
            }
            ffmpeg_checked = true;
        }
    }

    clock_t start_time = clock();
    int result = system(cmd);
    clock_t end_time = clock();

    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    if (result == 0)
    {
        std::cout << "Command executed successfully (" << std::fixed << std::setprecision(1) << elapsed_time << " seconds)" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Command failed with exit code: " << result << " (" << std::fixed << std::setprecision(1) << elapsed_time << " seconds)" << std::endl;
        return -1;
    }
}

// Validate that video files exist and are accessible
int validate_video_files(FileGroup *video_urls)
{
    if (!video_urls || video_urls->size() == 0)
    {
        std::cout << "No video files to validate" << std::endl;
        return 0;
    }

    int valid_files = 0;

    for (size_t i = 0; i < video_urls->size(); i++)
    {
        FILE *test_file = fopen(video_urls->filenames[i].c_str(), "rb");
        if (test_file)
        {
            fclose(test_file);
            valid_files++;
        }
        else
        {
            std::cout << video_urls->filenames[i] << " (file not accessible)" << std::endl;
        }
    }

    return valid_files;
}

/**
 * Create individual video cuts for each pump
 *
 * This function processes detected pump events and creates individual video clips
 * for each pump. Video quality is controlled by the Parameters.video_quality field:
 *
 * Video Quality Options:
 * - "low": Fast processing, smaller files (CRF 28, superfast preset)
 * - "medium": Balanced quality/speed (CRF 23, fast preset) [Default]
 * - "high": Better quality, slower processing (CRF 20, medium preset)
 * - "very high": Best quality, slowest processing (CRF 18, slow preset)
 * - "copy": No re-encoding, fastest but limited editing capability
 *
 * @param video_urls Pointer to FileGroup containing video file paths
 * @param pumps Array of detected Pump structures with timing information
 * @param pump_count Number of pump events to process
 * @param args Parameters structure containing configuration options
 */
void create_individual_cuts(const FileGroup *video_urls, const Pump *pumps, int pump_count, const Parameters &args, AppState *app_state)
{
    if (!video_urls || video_urls->size() == 0)
    {
        std::cout << "No video files available for processing" << std::endl;
        return;
    }

    // Convert all pump times to video offset seconds for processing
    std::vector<std::pair<int, double>> pump_starts; // pair<pump_index, start_seconds>
    for (int i = 0; i < pump_count; i++)
    {
        double pump_start_seconds = hhmmss_to_seconds(pumps[i].start_str);
        pump_starts.push_back({i, pump_start_seconds});
    }

    // Sort pumps by start time for efficient processing
    std::sort(pump_starts.begin(), pump_starts.end(),
              [](const auto &a, const auto &b)
              { return a.second < b.second; });

    // Single loop: accumulate video durations and process pumps as we go
    double accumulated_duration = 0.0;
    size_t next_pump_to_process = 0;

    std::cout << std::endl
              << "=== Cutting Video Files for GL" << video_urls->filenames[0].substr(video_urls->filenames[0].size()-8, 4) << " ===" << std::endl;

    for (size_t video_index = 0; video_index < video_urls->size(); ++video_index)
    {
        // Get current video file duration
        size_t mp4_handle = OpenMP4Source((char *)video_urls->filenames[video_index].c_str(), MOV_GPMF_TRAK_TYPE, MOV_GPMF_TRAK_SUBTYPE, 0);
        double current_file_duration = 0.0;

        if (mp4_handle)
        {
            current_file_duration = GetDuration(mp4_handle);
            CloseSource(mp4_handle);
            std::cout << std::endl
                      << "File " << (video_index + 1) << ": " << video_urls->filenames[video_index]
                      << " - Duration: " << std::fixed << std::setprecision(2) << current_file_duration
                      << "s, Accumulated: " << (accumulated_duration + current_file_duration) << "s" << std::endl;
        }
        else
        {
            std::cout << "Warning: Could not open file " << video_urls->filenames[video_index] << std::endl;
            continue; // Skip this file
        }
        // Process all pumps that fall within this video file's time range
        while (next_pump_to_process < pump_starts.size())
        {
            int pump_index = pump_starts[next_pump_to_process].first;
            double pump_start_seconds = pump_starts[next_pump_to_process].second;
            double pump_end_seconds = hhmmss_to_seconds(pumps[pump_index].end_str);

            // Check if pump starts within this video file
            if (pump_start_seconds >= accumulated_duration &&
                pump_start_seconds < accumulated_duration + current_file_duration)
            {

                std::cout << std::endl
                          << "Processing pump " << (pump_index + 1) << "/" << pump_count
                          << " - Start: " << pumps[pump_index].start_str << " (" << pump_start_seconds << "s)"
                          << " End: " << pumps[pump_index].end_str << " (" << pump_end_seconds << "s)"
                          << " Duration: " << std::fixed << std::setprecision(2) << pumps[pump_index].duration << "s" << std::endl;

                // Calculate pump times within this specific file
                double pump_start_in_file = pump_start_seconds - accumulated_duration;
                double pump_end_in_file = pump_end_seconds - accumulated_duration;

                // Clamp end time to file duration if it extends beyond
                if (pump_end_in_file > current_file_duration)
                {
                    pump_end_in_file = current_file_duration;
                    std::cout << "Note: Pump " << (pump_index + 1) << " extends beyond file "
                              << video_urls->filenames[video_index] << ". Trimming to file end." << std::endl;
                }

                // Add padding if requested
                if (args.add_10s)
                {
                    pump_start_in_file = (pump_start_in_file > 10.0) ? pump_start_in_file - 10.0 : 0.0;
                    pump_end_in_file = std::min(pump_end_in_file + 10.0, current_file_duration);
                }

                std::string start_time_str = secondsToHHMMSS((int)pump_start_in_file);
                std::string end_time_str = secondsToHHMMSS((int)pump_end_in_file);

                // Generate output filename
                char output_filename[512];
                snprintf(output_filename, sizeof(output_filename),
                         "GL%s/pump_%02d.mp4",
                         video_urls->filenames[video_index].substr(video_urls->filenames[video_index].size()-8, 4).c_str(), // Extract GL sequence from filename
                         pump_index + 1);

                // Get video encoding parameters based on quality level
                VideoEncodingParams encoding_params = get_video_encoding_params(args.video_quality);

                // Build FFmpeg command with quality-specific parameters
                char ffmpeg_cmd[2048]; // Increased buffer size for longer commands
                bool command_succeeded = false;

                if (args.add_timestamps)
                {
                    // Calculate duration instead of absolute end time
                    double duration = pump_end_in_file - pump_start_in_file;
                    char duration_str[16];
                    snprintf(duration_str, sizeof(duration_str), "%.2f", duration);

                    // Encoding with timestamps using quality-specific parameters
                    snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                             "ffmpeg -y -loglevel quiet -ss %s -i \"%s\" -t %s -avoid_negative_ts make_zero -vf \"drawtext=text='%%{pts\\:localtime\\:%.3f\\:%%Y-%%m-%%d %%H\\\\\\:%%M\\\\\\:%%S}':x=10:y=H-th-10:font=Arial:fontsize=24:fontcolor=white@1.0:box=1:boxcolor=black@0.5:boxborderw=5\" -c:v %s -preset %s -crf %d %s -c:a copy \"%s\"",
                             start_time_str.c_str(), video_urls->filenames[video_index].c_str(), duration_str,
                             pumps[pump_index].start_time, encoding_params.codec.c_str(), encoding_params.preset.c_str(),
                             encoding_params.crf, encoding_params.additional_params.c_str(), output_filename);
                    command_succeeded = (execute_ffmpeg_command(ffmpeg_cmd) == 0);
                }
                else
                {
                    // Calculate duration for non-timestamp path too
                    double duration = pump_end_in_file - pump_start_in_file;
                    char duration_str[16];
                    snprintf(duration_str, sizeof(duration_str), "%.2f", duration);

                    // Check if copy mode should be used (video_quality is "copy" or empty)
                    if (args.video_quality == "copy" || args.video_quality.empty())
                    {
                        snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                                 "ffmpeg -y -loglevel quiet -ss %s -i \"%s\" -t %s -c copy \"%s\"",
                                 start_time_str.c_str(), video_urls->filenames[video_index].c_str(), duration_str, output_filename);
                    }
                    else
                    {
                        // Re-encode with quality-specific parameters
                        snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
                                 "ffmpeg -y -loglevel quiet -ss %s -i \"%s\" -t %s -c:v %s -preset %s -crf %d %s -c:a copy \"%s\"",
                                 start_time_str.c_str(), video_urls->filenames[video_index].c_str(), duration_str,
                                 encoding_params.codec.c_str(), encoding_params.preset.c_str(),
                                 encoding_params.crf, encoding_params.additional_params.c_str(), output_filename);
                    }
                    command_succeeded = (execute_ffmpeg_command(ffmpeg_cmd) == 0);
                }

                // Check if command failed
                if (!command_succeeded)
                {
                    std::cout << "Failed to create cut " << (pump_index + 1) << std::endl;
                    
                }
                ++app_state->exported_videos;
                next_pump_to_process++;
            }
            else if (pump_start_seconds >= accumulated_duration + current_file_duration)
            {
                // This pump and all subsequent ones are in later video files
                break;
            }
            else
            {
                // This pump was in an earlier video file (shouldn't happen with sorted order)
                std::cout << "Warning: Pump " << (pump_index + 1) << " skipped - starts before current video file range." << std::endl;
                next_pump_to_process++;
            }
        }

        // Update accumulated duration for next iteration
        accumulated_duration += current_file_duration;
    }

    // Report any pumps that couldn't be processed
    if (next_pump_to_process < pump_starts.size())
    {
        std::cout << "\nWarning: " << (pump_starts.size() - next_pump_to_process)
                  << " pump(s) extend beyond total video duration ("
                  << std::fixed << std::setprecision(2) << accumulated_duration << "s) and were skipped." << std::endl;
        app_state->exported_videos += (pump_starts.size() - next_pump_to_process);
    }

}

// Create a concatenated supercut of all pumps
void create_concatenated_supercut(const FileGroup *video_urls, const vector<Pump> &pumps, const Parameters &args, AppState *app_state)
{
    if (pumps.empty()) {
        std::cout << "No pumps to concatenate" << std::endl;
        return;
    }

    // Get video encoding parameters based on quality level
    VideoEncodingParams encoding_params = get_video_encoding_params(args.video_quality);

    // Build multiple input arguments for FFmpeg (no temp file needed)
    std::string ffmpeg_inputs;
    std::string filter_complex = "\"";
    
    for (size_t i = 0; i < pumps.size(); i++)
    {
        char cut_filename[512];
        std::string output_path = args.output_dir.empty() ? "" : args.output_dir;
        if (!output_path.empty() && output_path.back() != '\\' && output_path.back() != '/') {
            output_path += "\\";
        }
        
        snprintf(cut_filename, sizeof(cut_filename),
                 "%sGL%s\\pump_%02lld.mp4",
                 output_path.c_str(),
                 video_urls->filenames[0].substr(video_urls->filenames[0].size()-8, 4).c_str(), i + 1);

        // Add input file
        ffmpeg_inputs += " -i \"";
        ffmpeg_inputs += cut_filename;
        ffmpeg_inputs += "\"";
        
        // Add to filter complex for concatenation
        if (i > 0) filter_complex += " ";
        filter_complex += "[" + std::to_string(i) + ":v] [" + std::to_string(i) + ":a]";
    }
    
    // Complete filter complex
    filter_complex += " concat=n=" + std::to_string(pumps.size()) + ":v=1:a=1 [v] [a]\"";

    // Create concatenated video
    char supercut_filename[256];
    std::string groupid = video_urls->filenames[0].substr(video_urls->filenames[0].size()-8, 4).c_str();
    
    // Ensure proper path separator for Windows
    std::string output_path = args.output_dir.empty() ? "" : args.output_dir;
    if (!output_path.empty() && output_path.back() != '\\' && output_path.back() != '/') {
        output_path += "\\";
    }
    
    snprintf(supercut_filename, sizeof(supercut_filename),
             "%sGL%s\\supercut_%s.mp4",
             output_path.c_str(),
             groupid.c_str(), groupid.c_str());

    // Build complete FFmpeg command with quality optimizations
    // Note: Stream copy cannot be used with complex filtergraphs (concat filter)
    // so we must re-encode both video AND audio streams
    std::string concat_cmd = "ffmpeg -y -loglevel quiet" + ffmpeg_inputs + 
                            " -filter_complex " + filter_complex + 
                            " -map \"[v]\" -map \"[a]\" -c:v " + encoding_params.codec + 
                            " -preset " + encoding_params.preset + 
                            " -crf " + std::to_string(encoding_params.crf) + 
                            " " + encoding_params.additional_params + 
                            " -c:a aac \"" + supercut_filename + "\"";

    std::cout << "Creating supercut with " << args.video_quality << " quality..." << std::endl;
    if (execute_ffmpeg_command(concat_cmd.c_str()) == 0)
    {
        // Calculate total duration
        double total_duration = 0;
        for (size_t i = 0; i < pumps.size(); i++)
        {
            total_duration += pumps[i].duration;
            if (args.add_10s)
                total_duration += 20.0; // 10s before + 10s after
        }
        
        std::cout << "Supercut created successfully. Total duration: " 
                  << std::fixed << std::setprecision(2) << total_duration << " seconds" << std::endl;
    }


    ++app_state->exported_videos;

}

void create_supercut(FileGroup *video_urls, vector<Pump> &pumps, Parameters args, AppState *app_state)
{
    if (!video_urls || pumps.empty())
    {
        std::cout << "Invalid parameters for supercut creation" << std::endl;
        return;
    }

    if (video_urls->size() == 0)
    {
        std::cout << "No video files available for cutting" << std::endl;
        return;
    }

    // Validate video files exist
    int valid_files = validate_video_files(video_urls);
    if (valid_files == 0)
    {
        std::cout << "Error: No valid video files found for processing" << std::endl;
        return;
    }

    // Create individual cuts if requested
    create_individual_cuts(video_urls, pumps.data(), pumps.size(), args, app_state);
    create_concatenated_supercut(video_urls, pumps, args, app_state);
}

vector<FileGroup> listAndSortFiles(const std::string &directory)
{
    struct FileEntry
    {
        std::string filename = "";
        int sequence = 0;
        std::string group_id = "";
    };

    map<std::string, vector<FileEntry>> grouped_files;

#ifdef _WIN32
    struct _finddata_t file_data;
    intptr_t handle;
    std::string search_path = directory + "\\*.*";

    handle = _findfirst(search_path.c_str(), &file_data);
    if (handle == -1)
    {
        return {};
    }

    do
    {
        std::string filename = file_data.name;
        if (isGoProFile(filename) && filename.length() >= 12)
        {
            // Extract XXX (sequence) and YYYY (group) from filename
            std::string xxx_str = filename.substr(2, 2);
            std::string yyy_str = filename.substr(4, 4);

            int sequence = stoi(xxx_str);

            FileEntry entry{filename, sequence, yyy_str};
            grouped_files[yyy_str].push_back(entry);
        }
    } while (_findnext(handle, &file_data) == 0);

    _findclose(handle);

#else
    DIR *dir = opendir(directory.c_str());
    if (!dir)
        return {};

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (isGoProFile(filename) && filename.length() >= 12)
        {
            std::string xxx_str = filename.substr(2, 2);
            std::string yyy_str = filename.substr(4, 4);

            int sequence = stoi(xxx_str);

            FileEntry file_entry{filename, sequence, yyy_str};
            grouped_files[yyy_str].push_back(file_entry);
        }
    }
    closedir(dir);
#endif

    // Create FileGroup and sort files within each group
    vector<FileGroup> file_groups;

    for (auto &[group_id, entries] : grouped_files)
    {
        // Sort by sequence number
        std::sort(entries.begin(), entries.end(),
                  [](const FileEntry &a, const FileEntry &b)
                  {
                      return a.sequence < b.sequence;
                  });

        // Create a FileGroup for this group
        FileGroup file_group;
        file_group.group_id = group_id;

        // Add all filenames to this FileGroup
        for (const auto &entry : entries)
        {
            file_group.addFile(entry.filename);
        }

        file_groups.push_back(std::move(file_group));
    }

    return file_groups;
}

// Create FileGroup and sort files within each group
vector<FileGroup> sortFilesByGroup(vector<std::string> filenames)
{
    vector<FileGroup> file_groups;

    // Group files by their group ID
    std::map<std::string, std::vector<FileEntry>> grouped_files;
    for (const auto &filename : filenames)
    {
        FileEntry entry;

        entry.filename = filename.c_str();
        std::string group_id = filename.substr(filename.length() - 8, 4);     // Extract group ID from filename
        int sequence = std::stoi(filename.substr(filename.length() - 10, 2)); // Extract sequence number
        entry.sequence = sequence;
        grouped_files[group_id].push_back(entry);
    }

    for (auto &[group_id, entries] : grouped_files)
    {
        // Sort by sequence number
        std::sort(entries.begin(), entries.end(),
                  [](const FileEntry &a, const FileEntry &b)
                  {
                      return a.sequence < b.sequence;
                  });

        // Create a FileGroup for this group
        FileGroup file_group;
        file_group.group_id = group_id;

        // Add all filenames to this FileGroup
        for (const auto &entry : entries)
        {
            file_group.addFile(entry.filename);
        }

        file_groups.push_back(std::move(file_group));
    }

    return file_groups;
}

std::string addZwsAfterPeriods(const std::string &in)
{
    std::string out;
    for (char c : in)
    {
        out += c;
        if (c == '.')
            out += "\xE2\x80\x8B"; // raw UTF-8 for U+200B
    }
    return out;
}

void wrapRenderText(const std::string &text, float wrap_width)
{
    const char *text_ptr = text.c_str();
    const char *end_ptr = text_ptr + text.size();
    ImFont *font = ImGui::GetFont();
    float font_size = ImGui::GetFontSize();

    ImVec2 base_cursor = ImGui::GetCursorScreenPos();
    float x = base_cursor.x;
    float y = base_cursor.y;

    while (text_ptr < end_ptr)
    {
        const char *word_end = text_ptr;
        const char *last_space = nullptr;

        while (word_end < end_ptr)
        {
            const char *next = word_end + 1;
            float width = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, text_ptr, next, nullptr).x;

            if (*word_end == ' ')
                last_space = word_end;

            if (width > wrap_width)
                break;

            word_end = next;
        }

        if (last_space && last_space > text_ptr && last_space < word_end)
            word_end = last_space;

        ImGui::SetCursorScreenPos(ImVec2(x, y));
        ImGui::TextUnformatted(text_ptr, word_end);

        y += ImGui::GetTextLineHeightWithSpacing(); // move down one line
        text_ptr = word_end;
        while (*text_ptr == ' ')
            ++text_ptr;
    }

    // Update final cursor Y so ImGui knows where we left off
    ImGui::SetCursorScreenPos(ImVec2(x, y));
}

// Function to check if all processing is complete - moved after AppState definition

class ParamsWindow
{
public:
    double *lowPass;
    double lowPassCutoff;
    AppState *app_state;

    ParamsWindow(AppState *state) : lowPass(nullptr), lowPassCutoff(0.0), app_state(state) {}
    ~ParamsWindow()
    {
        if (lowPass)
        {
            free(lowPass);
            lowPass = nullptr;
        }
    }

    void renderGUI(GLFWwindow *window);
};

class MenuWindow
{
private:
    const char *filenames = nullptr; // Pointer to hold the selected file path
    // Add any private members or methods if needed
public:
    AppState *app_state;

    MenuWindow(AppState *state) : app_state(state) {}
    ~MenuWindow()
    {
        // Note: tinyfiledialogs manages its own memory
        // Don't call free() on the pointer returned by tinyfd_openFileDialog
        filenames = nullptr;
    }

    void renderGUI(GLFWwindow *window);
};

void singlePlotWindow(double *x, double *y, int size, std::string title)
{
    vector<plot_data> plots;
    plot_data plotData;
    plotData.x = x;
    plotData.y = y;
    plotData.size = size;
    plotData.title = title.c_str();

    plots.push_back(plotData);
    multiPlottingWindow(plots);
}

// Function to check if all processing is complete
bool is_processing_complete(AppState *app_state)
{
    return app_state->processed_groups.load() >= app_state->total_groups.load();
}

// Function to get processing progress (0.0 to 1.0)
float get_processing_progress(AppState *app_state)
{
    int total = app_state->total_groups.load();
    if (total == 0)
        return 0.0f;

    int processed = app_state->processed_groups.load();
    return (float)processed / (float)total;
}

void PlotTwoLines(const char *label,
                  const float *data1, ImU32 color1,
                  const float *data2, ImU32 color2,
                  int count,
                  ImVec2 size = ImVec2(400, 200))
{
    if (!data1 || !data2 || count <= 1)
    {
        ImGui::Text("%s: Invalid data", label);
        return;
    }

    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();

    ImGui::Text("%s", label);
    ImGui::InvisibleButton(label, size); // Reserve the space

    // Draw frame background
    ImVec2 canvas_end = ImVec2(canvas_pos.x + size.x, canvas_pos.y + size.y);
    draw_list->AddRectFilled(canvas_pos, canvas_end, ImGui::GetColorU32(ImGuiCol_FrameBg), 5.0f);
    draw_list->AddRect(canvas_pos, canvas_end, ImGui::GetColorU32(ImGuiCol_Border));

    // Find min and max values for proper normalization
    float min_val = std::min(*std::min_element(data1, data1 + count),
                             *std::min_element(data2, data2 + count));
    float max_val = std::max(*std::max_element(data1, data1 + count),
                             *std::max_element(data2, data2 + count));

    // Avoid division by zero
    if (max_val == min_val)
    {
        max_val = min_val + 1.0f;
    }

    float range = max_val - min_val;

    for (int i = 1; i < count; ++i)
    {
        float x0 = (float)(i - 1) / (count - 1);
        float x1 = (float)(i) / (count - 1);

        // Normalize data to 0-1 range
        float y0a = (data1[i - 1] - min_val) / range;
        float y1a = (data1[i] - min_val) / range;

        float y0b = (data2[i - 1] - min_val) / range;
        float y1b = (data2[i] - min_val) / range;

        // Create normalized coordinates
        ImVec2 tp0a = ImVec2(x0, 1.0f - y0a);
        ImVec2 tp1a = ImVec2(x1, 1.0f - y1a);
        ImVec2 tp0b = ImVec2(x0, 1.0f - y0b);
        ImVec2 tp1b = ImVec2(x1, 1.0f - y1b);

        // Use ImLerp for smooth coordinate interpolation like ImGui does
        ImVec2 pos0a = ImLerp(canvas_pos, canvas_end, tp0a);
        ImVec2 pos1a = ImLerp(canvas_pos, canvas_end, tp1a);
        ImVec2 pos0b = ImLerp(canvas_pos, canvas_end, tp0b);
        ImVec2 pos1b = ImLerp(canvas_pos, canvas_end, tp1b);

        // Draw line segments
        draw_list->AddLine(pos0a, pos1a, color1, 1.5f);
        draw_list->AddLine(pos0b, pos1b, color2, 1.5f);
    }

    ImGui::SetCursorScreenPos(canvas_pos);
    ImGui::Text("%s", label);
}

void PlotTwoLinesGui(const char *label,
                     const float *data1, ImU32 color1,
                     const float *data2, ImU32 color2,
                     int count,
                     AppState *app_state = nullptr,
                     ImVec2 size = ImVec2(400, 200))
{
    if (!data1 || !data2 || count <= 1)
    {
        ImGui::Text("%s: Invalid data", label);
        return;
    }

    // Find min and max values for both datasets for consistent scaling
    float min_val = std::min(*std::min_element(data1, data1 + count),
                             *std::min_element(data2, data2 + count));
    float max_val = std::max(*std::max_element(data1, data1 + count),
                             *std::max_element(data2, data2 + count));

    // Avoid division by zero
    if (max_val == min_val)
    {
        max_val = min_val + 1.0f;
    }

    // Get the current cursor position before plotting
    ImVec2 cursor_pos = ImGui::GetCursorScreenPos();

    // Use ImGui's native PlotLines for the first line (clean rendering)
    ImGui::PlotLines("##plot", data1, count, 0, label, min_val, max_val, size);

    // Store if the plot was interacted with BEFORE checking popups
    bool plot_clicked = ImGui::IsItemClicked();
    bool plot_hovered = ImGui::IsItemHovered();
    bool mouse_down = ImGui::IsMouseDown(0);

    // Get the plot area info after PlotLines is rendered
    ImDrawList *draw_list = ImGui::GetWindowDrawList();

    // Calculate the actual plot area (PlotLines adds padding)
    const ImGuiStyle &style = ImGui::GetStyle();
    ImVec2 frame_padding = style.FramePadding;
    ImVec2 plot_min = ImVec2(cursor_pos.x + frame_padding.x, cursor_pos.y + frame_padding.y);
    ImVec2 plot_max = ImVec2(cursor_pos.x + size.x - frame_padding.x, cursor_pos.y + size.y - frame_padding.y);

    // Manually draw the second line on top
    float range = max_val - min_val;
    float plot_width = plot_max.x - plot_min.x;
    float plot_height = plot_max.y - plot_min.y;

    for (int i = 1; i < count; ++i)
    {
        // Calculate x positions
        float x0 = plot_min.x + ((float)(i - 1) / (count - 1)) * plot_width;
        float x1 = plot_min.x + ((float)(i) / (count - 1)) * plot_width;

        // Calculate y positions (normalized and flipped)
        float y0 = plot_max.y - ((data2[i - 1] - min_val) / range) * plot_height;
        float y1 = plot_max.y - ((data2[i] - min_val) / range) * plot_height;

        // Draw the line segment
        draw_list->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), color2, 1.5f);
    }

    // Add click and drag detection for setting pump threshold
    // ONLY process interaction if no popup/combo is active
    if (app_state &&
        // Check if any popup is open (most reliable check)
        !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId) &&
        // Check if any item is currently active or focused
        // !ImGui::IsAnyItemActive() &&
        // !ImGui::IsAnyItemFocused() &&
        // Check for plot interaction - only on actual click, not hover
        (plot_clicked || (plot_hovered && mouse_down)))
    {
        ImVec2 mouse_pos = ImGui::GetMousePos();

        // Check if click is within the plot area
        if (mouse_pos.x >= plot_min.x && mouse_pos.x <= plot_max.x &&
            mouse_pos.y >= plot_min.y && mouse_pos.y <= plot_max.y)
        {

            // Convert mouse y position back to data value
            float normalized_y = (plot_max.y - mouse_pos.y) / plot_height;
            float clicked_value = min_val + (normalized_y * range);

            // Update the pump threshold
            app_state->params.pump_thresh = (double)clicked_value;
        }
    }
}

void AppState::saveConfig() {
    std::ofstream out("params.txt");
    if (!out.is_open()) {
        std::cerr << "Error: Could not open params.txt for writing." << std::endl;
        return;
    }
    out << "accelCutOff: " << params.accel_cutoff << "\n";
    out << "lowPassCutoff: " << params.lowpass_cutoff << "\n";
    out << "fs: " << params.fs << "\n";
    out << "kernel_n (sec): " << params.kernel_n_sec << "\n";
    out << "pumpThresh: " << params.pump_thresh << "\n";
    out << "fine_kernel_n (sec): " << params.fine_kernel_n_sec << "\n";
    out << "plots: " << (params.plots ? "True" : "False") << "\n";
    out << "finePumpThresh: " << params.fine_pump_thresh << "\n";
    out << "velocityThresh: " << params.velocity_thresh << "\n";
    out << "addTimestamps: " << (params.add_timestamps ? "True" : "False") << "\n";
    out << "add10sec: " << (params.add_10s ? "True" : "False") << "\n";
    out << "videoQuality: " << params.video_quality << "\n";
    out << "outputDir: " << params.output_dir << "\n";
    out << "inputDir: " << params.input_dir << "\n";
    out.close();
}

void AppState::resetToDefaults() {
    params.accel_cutoff = 35.0;
    params.lowpass_cutoff = .4;
    params.fs = 201;
    params.kernel_n_sec = 15;
    params.pump_thresh = 1.7;
    params.fine_kernel_n_sec = 5;
    params.plots = false;
    params.fine_pump_thresh = 0.15;
    params.velocity_thresh = 4.0;
    params.add_timestamps = true;
    params.add_10s = false;
    params.video_quality = "medium";
    std::string default_output_dir = std::filesystem::current_path().string();
    params.output_dir = default_output_dir;
}

// AppState destructor implementation
AppState::~AppState()
{
    std::cout << "AppState destructor: Starting cleanup..." << std::endl;

    // Signal all threads to exit immediately
    should_exit = true;

    // Stop accepting new processing requests
    processing_active = false;

    // Wait for processing threads to complete with a reasonable timeout
    std::cout << "AppState destructor: Waiting for processing threads..." << std::endl;
    // const auto cleanup_start = std::chrono::steady_clock::now();
    const auto max_cleanup_time = std::chrono::seconds(30);

    {
        std::lock_guard<std::mutex> lock(futures_mutex);
        for (auto &future : processing_futures)
        {
            if (future.valid())
            {
                try
                {
                    auto wait_result = future.wait_for(std::chrono::seconds(5));
                    if (wait_result == std::future_status::ready)
                    {
                        future.get(); // Clean up the future
                    }
                    else
                    {
                        std::cerr << "Warning: Processing thread did not complete within timeout during destructor" << std::endl;
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Exception in destructor while waiting for thread: " << e.what() << std::endl;
                }
                catch (...)
                {
                    std::cerr << "Unknown exception in destructor while waiting for thread" << std::endl;
                }
            }
        }
        processing_futures.clear();
    }

    // Then wait for rendering thread
    std::cout << "AppState destructor: Waiting for rendering thread..." << std::endl;
    if (rendering_thread.joinable())
    {
        try
        {
            rendering_thread.join();
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception joining rendering thread: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception joining rendering thread" << std::endl;
        }
    }

    // Finally, cleanup window objects
    std::cout << "AppState destructor: Cleaning up windows..." << std::endl;
    if (params_window)
    {
        try
        {
            delete params_window;
            params_window = nullptr;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception deleting params_window: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception deleting params_window" << std::endl;
        }
    }
    if (menu_window)
    {
        try
        {
            delete menu_window;
            menu_window = nullptr;
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception deleting menu_window: " << e.what() << std::endl;
        }
        catch (...)
        {
            std::cerr << "Unknown exception deleting menu_window" << std::endl;
        }
    }

    std::cout << "AppState destructor: Cleanup complete." << std::endl;
}

void ParamsWindow::renderGUI(GLFWwindow *window)
{
    if (!app_state->show_params_window)
        return;

    // Center the parameters window on first use
    ImGuiIO &io = ImGui::GetIO();
    ImVec2 center = ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
    ImGui::SetNextWindowPos(center, 0, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(850.f,0));

    ImGui::SetNextWindowBgAlpha(1.0f); // Make it slightly more opaque to stand out

    if (ImGui::Begin("Parameters", &app_state->show_params_window,
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse))

    {
        ImVec2 win_size = ImGui::GetWindowSize();
        ImVec2 win_padding = ImGui::GetStyle().WindowPadding;

        ImGui::Text("Pump Detection Parameters:");
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(.5f, 1.0f, 0.5f, 1.0f));
        ImGui::TextWrapped("Drag to adjust the threshold to fine-tune pump detection. Anything above the threshold will be considered a pump.");
        ImGui::PopStyleColor();

        // Check if we have pump results to display
        if (!app_state->pump_results.empty() && !app_state->pump_results[0].signal.empty())
        {
            vector<double> data = app_state->pump_results[0].signal;
            vector<float> data_f(data.size());
            vector<float> threshold_line(data.size());
            float max_val = 0.0f;
            float min_val = 1.0f;
            for (size_t i = 0; i < data.size(); i++)
            {
                data_f[i] = static_cast<float>(data[i]);
                if (data_f[i] > max_val)
                    max_val = data_f[i];
                if (data_f[i] < min_val)
                    min_val = data_f[i];
                threshold_line[i] = app_state->params.pump_thresh;
            }
            ImU32 color = IM_COL32(200, 200, 200, 255);
            ImU32 threshold_color = IM_COL32(220, 120, 120, 255);
            ImVec2 plot_size = ImVec2(win_size.x - 2.0* win_padding.x, 120.f);

            std::string buffer = "Acceleration over Time for " + app_state->file_groups[0].group_id;
            PlotTwoLinesGui(buffer.c_str(), data_f.data(), color, threshold_line.data(), threshold_color, data_f.size(), app_state, plot_size);
        }
        else
        {
            // Show placeholder when no data is available
            ImGui::Text("Plot will appear here after processing video data.");
        }
        float stuffWidth = 400.0f; // Width for the stuff on the left side
        float halfwidth = (win_size.x - win_padding.x) - stuffWidth; // Half width for the right side

        float pump_thresh = (float)app_state->params.pump_thresh;
        ImGui::Text("Pump Threshold: ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);

        ImGui::PushItemWidth(stuffWidth);
        ImGui::InputFloat("##Pump Threshold", &pump_thresh, 0.01f, 0.1f, "%.3f");
        app_state->params.pump_thresh = (double)pump_thresh; // Update the parameter
        ImGui::PopItemWidth();
        ImGui::Separator();

        ImGui::Text("Video Settings:");

        char buffer[256] = "";
        float browseWidth = ImGui::CalcTextSize("Browse").x + ImGui::GetStyle().FramePadding.x * 2.0f;
        snprintf(buffer, sizeof(buffer), "%s", app_state->params.output_dir.c_str());
        ImGui::Text("Output Folder: ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        ImGui::PushItemWidth(stuffWidth - browseWidth - ImGui::GetStyle().ItemSpacing.x);
        ImGui::InputText("##Output Folder", buffer, sizeof(buffer),ImGuiInputTextFlags_AlwaysOverwrite|ImGuiInputTextFlags_ElideLeft);
        if (std::string(buffer) != app_state->params.output_dir && !std::string(buffer).empty()) {
            app_state->params.output_dir = buffer;
        }
        ImGui::PopItemWidth();
        ImGui::SameLine();
        if (ImGui::Button("Browse")) {
            std::string out;
            std::string wd = std::filesystem::current_path().string();
#ifdef _WIN32
            out = OpenFolderDialog_stdstring("Select Output Directory",app_state->params.output_dir.empty()? wd.c_str():app_state->params.output_dir.c_str());
#else
            out = tinyfd_selectFolderDialog("Select Output Directory", wd.c_str());                
#endif


            if (!out.empty()) {
                app_state->params.output_dir = out; // Update the output directory
            }
        }


        ImGui::Text("Video Quality: ");
        ImGui::SameLine();
        if (ImGui::Button("Help"))
        {
            ImGui::OpenPopup("HelpPopup");
        }
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        ImGui::PushItemWidth(stuffWidth);
        const char *video_quality_options[] = {"Low", "Medium", "High", "Very High"};
        if (ImGui::BeginCombo("##VideoQualityCombo", video_quality_options[app_state->params.vidQualityNum]))
        {
            for (int n = 0; n < IM_ARRAYSIZE(video_quality_options); n++)
            {
                bool is_selected = (app_state->params.vidQualityNum == n);
                if (ImGui::Selectable(video_quality_options[n], is_selected))
                {
                    app_state->params.vidQualityNum = n;
                    app_state->params.video_quality = video_quality_options[n];
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();



        // Help popup with useful information
        if (ImGui::BeginPopup("HelpPopup"))
        {
            ImGui::Text("Video Quality Settings:");
            ImGui::BulletText("Low: Fast encoding, smaller files");
            ImGui::BulletText("Medium: Balanced quality and speed (recommended)");
            ImGui::BulletText("High: Better quality, slower encoding");
            ImGui::BulletText("Very High: Best quality, slowest encoding");

            ImGui::Separator();
            ImGui::Text("Additional Options:");
            ImGui::BulletText("Timestamps: Add time overlay to videos");
            ImGui::BulletText("10s Padding: Add 10 seconds before/after each pump");

            ImGui::Separator();
            if (ImGui::Button("Close"))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
        if (ImGui::BeginPopupModal("Reset Confirmation", NULL, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove))
        {
            ImGui::Text("Are you sure you want to reset all parameters to defaults?");
            ImGui::Text("This will overwrite your current settings.");
            ImGui::Separator();
            if (ImGui::Button("Yes", ImVec2(120, 0)))
            {
                app_state->resetToDefaults();
                ImGui::CloseCurrentPopup();
            }
            ImGui::SameLine();
            if (ImGui::Button("No", ImVec2(120, 0)))
            {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }

        ImGui::PushItemWidth(halfwidth);
        ImGui::Checkbox("Add Timestamps to Videos", &app_state->params.add_timestamps);
        ImGui::PopItemWidth();
        ImGui::SameLine();
        ImGui::Checkbox("Add 10s Padding to Videos", &app_state->params.add_10s);



        ImGui::Separator();
        
        ImGui::Text("Data Processing Parameters:");
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.5f, 0.5f, 1.0f));
        ImGui::Text("(Don't change unless you're sure!)");
        ImGui::PopStyleColor();

        ImGui::PushItemWidth(stuffWidth);
        // Use InputFloat and InputInt instead of sliders
        ImGui::Text("Sample Rate (Hz): ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        ImGui::InputInt("##Samplerate", &app_state->params.fs, 1, 20);


        ImGui::Text("Low Pass Cutoff (rad/sample*s): ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        float low_pass_cutoff = (float)app_state->params.lowpass_cutoff;
        ImGui::InputFloat("##Low Pass Cutoff", &low_pass_cutoff, 0.01f, 1.0f, "%.2f");
        app_state->params.lowpass_cutoff = (double)low_pass_cutoff; // Update the parameter


        ImGui::Text("Acceleration Cutoff (m/s^2): ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        float accel_cutoff = (float)app_state->params.accel_cutoff;
        ImGui::InputFloat("##Acceleration Cutoff", &accel_cutoff, .5f, 5.0f, "%.1f");
        app_state->params.accel_cutoff = (double)accel_cutoff; // Update the parameter


        ImGui::Text("Moving Average Size (Seconds): ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        ImGui::InputInt("##Moving Average Size (Seconds)", &app_state->params.kernel_n_sec, 1, 10);
        if (app_state->params.kernel_n_sec < 1)
            app_state->params.kernel_n_sec = 1; // Ensure minimum size

        ImGui::Text("Small Moving Average Size (Seconds): ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        ImGui::InputInt("##Small Moving Average Size (Seconds)", &app_state->params.fine_kernel_n_sec, 1, 10);
        if (app_state->params.fine_kernel_n_sec < 1)
            app_state->params.fine_kernel_n_sec = 1; // Ensure minimum size

        ImGui::Text("Pump Refining Threshold (m/s^2): ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        float pump_refining_threshold = (float)app_state->params.fine_pump_thresh;
        ImGui::InputFloat("##Pump Refining Threshold", &pump_refining_threshold, .01f, 0.05f, "%.2f");
        app_state->params.fine_pump_thresh = (double)pump_refining_threshold; // Update the parameter

        ImGui::Text("Velocity Threshold (m/s): ");
        ImGui::SameLine();
        ImGui::SetCursorPosX(halfwidth);
        float velocity_cutoff = (float)app_state->params.velocity_thresh;
        ImGui::InputFloat("##Velocity Threshold (m/s)", &velocity_cutoff, 0.1f, 1.0f, "%.1f");
        app_state->params.velocity_thresh = (double)velocity_cutoff; // Update the parameter

        ImGui::PopItemWidth();
        // Add some helpful text for valid ranges
        ImGui::Separator();

        if(ImGui::Button("Save Config")) {
            app_state->saveConfig();
            start_processing_threads(app_state);
            app_state->show_params_window = false; // Close the window after saving
        }

        ImGui::SameLine();

        if(ImGui::Button("Reset to Defaults")) {
            ImGui::OpenPopup("Reset Confirmation");
        }
    }
    ImGui::End();
}

void MenuWindow::renderGUI(GLFWwindow *window)
{
    // Static cursors for hover effect
    static GLFWcursor *hand_cursor = nullptr;
    static GLFWcursor *arrow_cursor = nullptr;
    static bool cursors_initialized = false;

    // Initialize cursors on first call
    if (!cursors_initialized)
    {
        hand_cursor = glfwCreateStandardCursor(GLFW_HAND_CURSOR);
        arrow_cursor = glfwCreateStandardCursor(GLFW_ARROW_CURSOR);
        cursors_initialized = true;
    }

    // Use ImGui's display size which automatically updates during resize
    ImGuiIO &io = ImGui::GetIO();
    float display_w = io.DisplaySize.x;
    float display_h = io.DisplaySize.y;

    // Create main window that fills the entire screen
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(display_w, display_h), ImGuiCond_Always);

    ImGui::Begin("GoPro Pump Detector", NULL,
                 ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove |
                     ImGuiWindowFlags_NoTitleBar |
                     ImGuiWindowFlags_NoScrollbar |
                     ImGuiWindowFlags_NoBringToFrontOnFocus);

    // Center the content
    float center_x = display_w * 0.5f;
    float center_y = display_h * 0.5f;

    const char *title = "GoPro Video Pump Parser for Windsurfing";

    float titlewidth = ImGui::CalcTextSize(title).x;

    float content_width = std::max(display_w * 0.6f, titlewidth + 20.0f); // Limit width to 300px max
    float content_height = std::max(display_h * 0.6f, 500.0f);

    ImGui::SetCursorPos(ImVec2(center_x - content_width * 0.5f, center_y - content_height * 0.5f));

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.1f, 0.1f, 0.3f, 0.9f));
    ImGui::BeginChild("main_content", ImVec2(content_width, content_height), true, ImGuiWindowFlags_NoScrollbar);

    // Title with larger font
    ImGui::PushFont(ImGui::GetIO().FontDefault);
    ImGui::SetCursorPosX((content_width - titlewidth) * 0.5f);
    ImGui::Text(title);
    ImGui::PopFont();

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();
    if (app_state->filenames)
    {
        ImGui::Dummy(ImVec2(0, ImGui::GetTextLineHeight())); // Add some space before the file list

    }
    else
    {
        const char *no_files_text = "No files selected yet.";
        ImGui::SetCursorPosX((content_width - ImGui::CalcTextSize(no_files_text).x) * 0.5f);
        ImGui::TextWrapped("%s", no_files_text);
    }
    ImGui::Spacing();
    // Center the button
    float button_width = 300.0f;
    ImGui::SetCursorPosX((content_width - button_width) * 0.5f);

    const char *const validPatterns[] = {"*.LRV", "*.mp4"};

    if (ImGui::Button("Upload GoPro Videos", ImVec2(button_width, 50.0f)))
    {
        const char* result = tinyfd_openFileDialog("Select GoPro Videos", app_state->params.input_dir.c_str(), 2, validPatterns, NULL, 1);
        if (result)
        {
            app_state->filenames = result;
            std::vector<std::string> filenames_vector;
            std::string_view sv(app_state->filenames); // No copy

            size_t start = 0;
            while (true)
            {
                size_t pos = sv.find('|', start);
                if (pos == std::string_view::npos)
                {
                    filenames_vector.emplace_back(sv.substr(start));
                    break;
                }
                filenames_vector.emplace_back(sv.substr(start, pos - start));
                start = pos + 1;
            }

            app_state->params.input_dir = std::filesystem::path(filenames_vector[0]).string();
            app_state->file_groups = sortFilesByGroup(filenames_vector);
            // start processing each video group in a separate thread, and save each one's result to app_state->pump_results
            start_processing_threads(app_state);
        }
    }

    // Track if any button is hovered for cursor management
    bool upload_button_hovered = ImGui::IsItemHovered();
    bool config_button_hovered = false;
    bool extract_button_hovered = false;

    ImGui::Spacing();
    ImGui::Spacing();

    // Configure Parameters button - disabled during processing
    ImGui::SetCursorPosX((content_width - button_width) * 0.5f);

    bool is_processing = app_state->processing_active && !is_processing_complete(app_state);

    // Disable button if processing is active
    bool should_disable_config = is_processing || !app_state->filenames;
    if (should_disable_config)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
    }

    if (ImGui::Button("Configure Parameters", ImVec2(button_width, 50.0f)))
    {
        if (!should_disable_config)
        {
            // Open the parameters window
            std::cout << "Opening parameters window..." << std::endl;
            app_state->show_params_window = true;
        }
    }

    if (!should_disable_config)
    {
        // Only set hovered state if button is enabled
        config_button_hovered = ImGui::IsItemHovered();
    }
    else
    {
        ImGui::PopStyleColor(3);
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // Extract Pumps button - disabled during processing and when no files selected
    ImGui::SetCursorPosX((content_width - button_width) * 0.5f);

    bool should_disable_extract = !app_state->filenames || is_processing;

    if (should_disable_extract)
    {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.2f, 0.2f, 0.2f, 1.0f));
    }

    if (ImGui::Button("Export Videos and CSV", ImVec2(button_width, 50.0f)))
    {
        if (app_state->filenames && !is_processing)
        {
            start_exporting_threads(app_state);
        }
    }

    if (!should_disable_extract)
    {
        extract_button_hovered = ImGui::IsItemHovered();
    }

    if (should_disable_extract)
    {
        ImGui::PopStyleColor(3);
    }

    ImGui::Spacing();
    ImGui::Spacing();

    // Independent progress bar - show when processing is active or recently completed
    if (app_state->processing_active || app_state->cleanup_done)
    {

        // Show progress bar during and after processing
        float progress = get_processing_progress(app_state);
        bool is_complete = is_processing_complete(app_state);


        char progress_text[256];
        float buttonHeight = 28.0f;
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 20.0f);
        if (is_complete)
        {

            // Processing completed
            ImGui::SetCursorPosX((content_width - button_width) * 0.5f);
            ImVec4 green = ImVec4(0.0f, 0.7f, 0.0f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, green);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, green);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, green);
            ImGui::Button("Uploading Complete!", ImVec2(button_width, buttonHeight));
            ImGui::PopStyleColor(3);

            // Show results summary
            ImGui::Spacing();
            int total_pumps = 0;
            {
                std::lock_guard<std::mutex> lock(app_state->results_mutex);
                for (const auto &result : app_state->pump_results)
                {
                    total_pumps += (int)result.pumps.size();
                }
            }

            char results_text[256];
            snprintf(results_text, sizeof(results_text), "Found %d pumps across %d video groups",
                     total_pumps, app_state->total_groups.load());
            float text_width = ImGui::CalcTextSize(results_text).x;
            ImGui::SetCursorPosX((content_width - text_width) * 0.5f);
            ImGui::Text("%s", results_text);
        }
        else
        {
            ImGui::SetCursorPosX((content_width - button_width) * 0.5f);
            //ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.8f, 0.8f, 0.1f, 1.0f));
            ImGui::ProgressBar(progress, ImVec2(button_width, buttonHeight));
            //ImGui::PopStyleColor();
            snprintf(progress_text, sizeof(progress_text), "Processing: %d / %d groups (%.1f%%)",
                     app_state->processed_groups.load(),
                     app_state->total_groups.load(),
                     progress * 100.0f);
            float text_width = ImGui::CalcTextSize(progress_text).x;
            ImGui::SetCursorPosX((content_width - text_width) * 0.5f);
            ImGui::Text("%s", progress_text);
        }
        ImGui::PopStyleVar();
        ImGui::Spacing();
    } else if (app_state->export_active) {
        // Show export progress bar
        float export_progress;
        if (app_state->total_videos.load() == 0)
            export_progress = 1.0f;
        else
            export_progress = (float)app_state->exported_videos.load() / (float)app_state->total_videos.load();
        char export_text[256];
        float text_width;
        ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 20.0f);
        ImGui::SetCursorPosX((content_width - button_width) * 0.5f);
        if (export_progress >= 1.0f)
        {
            // Export completed
            ImVec4 green = ImVec4(0.0f, 0.7f, 0.0f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Button, green);
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, green);
            ImGui::PushStyleColor(ImGuiCol_ButtonActive, green);
            ImGui::Button("Export Complete!", ImVec2(button_width, 28.0f));
            ImGui::PopStyleColor(3);


            snprintf(export_text, sizeof(export_text), "Exported %d Pumping videos successfully",
                     app_state->exported_videos.load());
            text_width = ImGui::CalcTextSize(export_text).x;
        }
        else
        {
            //ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ImVec4(0.8f, 0.8f, 0.1f, 1.0f));
            ImGui::ProgressBar(export_progress, ImVec2(button_width, 28.0f));
            //ImGui::PopStyleColor();
            snprintf(export_text, sizeof(export_text), "Exporting: %d / %d videos (%.1f%%)",
                        app_state->exported_videos.load(),
                        app_state->total_videos.load(),
                        export_progress * 100.0f);
            text_width = ImGui::CalcTextSize(export_text).x;
        }
        ImGui::SetCursorPosX((content_width - text_width) * 0.5f);
        ImGui::Text("%s", export_text);
        ImGui::PopStyleVar();
    }

    // Processing completion handling
    if (app_state->processing_active && is_processing_complete(app_state))
    {
        // Processing completed - start cleanup thread if not already started
        // Use atomic compare-and-swap to prevent multiple threads from being created
        bool expected = false;
        if (!app_state->cleanup_done &&
            app_state->cleanup_thread_started.compare_exchange_strong(expected, true))
        {

            // Only one thread will succeed in the compare_exchange_strong operation
            // Capture the app_state pointer for the lambda
            AppState *state_ptr = app_state;

            // Start cleanup in a separate thread to avoid blocking UI
            std::thread cleanup_thread([state_ptr]()
                                       {
                std::cout << "Cleanup thread: Starting cleanup of completed processing threads..." << std::endl;
                try {
                    wait_for_processing_completion(state_ptr);
                    
                    // Reset processing state
                    state_ptr->processing_active = false;
                    state_ptr->cleanup_done = true;
                    
                    std::cout << "Cleanup thread: All threads cleaned up successfully." << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Cleanup thread: Exception during cleanup: " << e.what() << std::endl;
                    state_ptr->cleanup_done = true; // Set to true to avoid infinite loop
                    state_ptr->processing_active = false;
                } catch (...) {
                    std::cerr << "Cleanup thread: Unknown exception during cleanup" << std::endl;
                    state_ptr->cleanup_done = true; // Set to true to avoid infinite loop
                    state_ptr->processing_active = false;
                } });

            // Detach the cleanup thread so it runs independently
            cleanup_thread.detach();
        }
    }

    // Centralized cursor management - set cursor once based on any button hover
    // This runs regardless of which UI state we're in
    if (upload_button_hovered || config_button_hovered || extract_button_hovered)
    {
        glfwSetCursor(window, hand_cursor);
    }
    else
    {
        glfwSetCursor(window, arrow_cursor);
    }

    ImGui::EndChild();
    ImGui::PopStyleColor();

    ImGui::End();
}

// Forward declare the render function
void render_frame(GLFWwindow *window, AppState &app_state);

// Window resize callback that updates viewport
void window_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);

    // Don't force immediate redraw to avoid ImGui frame management conflicts
    // The rendering thread will handle the next frame automatically
}

// Render function that can be called from main loop or resize callback
void render_frame(GLFWwindow *window, AppState &app_state)
{
    if (!window || app_state.should_exit)
    {
        return;
    }

    try
    {
        // Start ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Render all windows with null checks
        if (app_state.params_window)
        {
            app_state.params_window->renderGUI(window);
        }
        if (app_state.menu_window)
        {
            app_state.menu_window->renderGUI(window);
        }

        // End ImGui frame and render
        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in render_frame: " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown exception in render_frame" << std::endl;
    }
}

// Threading function for rendering
void rendering_thread_function(GLFWwindow *window, AppState *app_state)
{
    if (!window || !app_state)
    {
        std::cerr << "Invalid parameters passed to rendering thread" << std::endl;
        return;
    }

    // OpenGL context must be current in this thread
    glfwMakeContextCurrent(window);

    // Verify OpenGL context is valid
    if (!glfwGetCurrentContext())
    {
        std::cerr << "Failed to make OpenGL context current in rendering thread" << std::endl;
        return;
    }

    const double target_fps = 60.0;
    const double frame_time = 1.0 / target_fps;

    std::cout << "Rendering thread started successfully" << std::endl;

    while (!app_state->should_exit && !glfwWindowShouldClose(window))
    {
        try
        {
            auto frame_start = std::chrono::high_resolution_clock::now();

            // Check if context is still valid
            if (!glfwGetCurrentContext())
            {
                std::cerr << "OpenGL context lost in rendering thread" << std::endl;
                break;
            }

            // Render the frame
            render_frame(window, *app_state);

            // Check for OpenGL errors
            GLenum error = glGetError();
            if (error != GL_NO_ERROR)
            {
                std::cerr << "OpenGL error in rendering thread: " << error << std::endl;
            }

            glfwSwapBuffers(window);

            // Calculate frame timing
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_duration = std::chrono::duration<double>(frame_end - frame_start).count();

            // Sleep to maintain target framerate
            if (frame_duration < frame_time)
            {
                std::this_thread::sleep_for(
                    std::chrono::duration<double>(frame_time - frame_duration));
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception in rendering thread: " << e.what() << std::endl;
            // Don't break immediately - try to continue rendering
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps fallback
        }
        catch (...)
        {
            std::cerr << "Unknown exception in rendering thread" << std::endl;
            // Don't break immediately - try to continue rendering
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps fallback
        }
    }

    // Release OpenGL context before thread exits
    glfwMakeContextCurrent(nullptr);

    std::cout << "Rendering thread exiting normally" << std::endl;
}

int main()
{

#if defined(IMGUI_IMPL_OPENGL_ES2)
    // GL ES 2.0 + GLSL 100 (WebGL 1.0)
    const char *glsl_version = "#version 100";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(IMGUI_IMPL_OPENGL_ES3)
    // GL ES 3.0 + GLSL 300 es (WebGL 2.0)
    const char *glsl_version = "#version 300 es";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
    // GL 3.2 + GLSL 150
    const char *glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow *window = glfwCreateWindow(1280, 880, "GoPro Video Pump Detector", NULL, NULL);
    if (!window)
    {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Set the window resize callback
    glfwSetWindowSizeCallback(window, window_size_callback);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();
    ImGuiStyle &style = ImGui::GetStyle();
    style.WindowRounding = 5.0f;
    style.FrameRounding = 3.0f;
    style.ChildRounding = 5.0f;

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load custom font with error checking
    ImFont *font = io.Fonts->AddFontFromFileTTF("C:/Windows/Fonts/arial.ttf", 24.0f);
    if (!font)
    {
        std::cout << "Warning: Could not load arial.ttf, using default font" << std::endl;
    }

    // Create shared application state
    AppState app_state = {};
    app_state.params = parse_params("params.txt"); // Initialize with default params

    // Initialize window pointers with error checking
    try
    {
        app_state.params_window = new ParamsWindow(&app_state);
        app_state.menu_window = new MenuWindow(&app_state);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Failed to create window objects: " << e.what() << std::endl;
        glfwTerminate();
        return -1;
    }

    // Set the app_state object as user pointer so callbacks can access it
    glfwSetWindowUserPointer(window, &app_state);

    // Start rendering thread
    // Note: We need to release the OpenGL context from main thread first
    glfwMakeContextCurrent(nullptr);
    app_state.rendering_thread = std::thread(rendering_thread_function, window, &app_state);

    // Main thread handles events only
    while (!glfwWindowShouldClose(window) && !app_state.should_exit)
    {
        glfwPollEvents();

        // Small sleep to prevent busy waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Signal rendering thread to exit
    app_state.should_exit = true;

    // Wait for rendering thread to finish
    if (app_state.rendering_thread.joinable())
    {
        app_state.rendering_thread.join();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
