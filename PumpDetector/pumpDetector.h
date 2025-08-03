#define MAX_FILES 100
#define MAX_FILENAME_LEN 256
#define MAX_PATH_LEN 512
#define MAX_PUMPS 1000
#define MAX_DATA_POINTS 1000000
#define DSPL_ERROR_PTR -1

#include "types.h"

PumpDetectionResult main_part(FileGroup* filenames, Parameters params);
std::string secondsToHHMMSS(int total_seconds);