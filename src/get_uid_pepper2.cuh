#ifndef GET_UID_PEPPER2_CUH
#define GET_UID_PEPPER2_CUH

#include <fstream>
#include <sstream>

#include "pepper.cuh"

#define FILENAME "./uid_pepper"

void get_uid_pepper2(char* out, const char* filename = FILENAME);

#endif