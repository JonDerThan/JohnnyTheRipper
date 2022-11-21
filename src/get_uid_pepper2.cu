#include "get_uid_pepper2.cuh"

void get_uid_pepper2(char* out, const char* filename) {
	std::ifstream file{filename};

	std::string line;
	while (std::getline(file, line)) {
		// copy line to out
		int i;
		for (i = 1; i <= line.length(); i++)
			out[PEPPER1_START - i] = line[line.length() - i];

		// set leading characters to _
		while (i <= PEPPER1_START)
			out[PEPPER1_START - i++] = '_';

		out += MAX_STRING_LEN;
	}
}