#include "stdafx.h"

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Incorrect number of input arguments, please enter filename"<< std::endl;
		exit(1);
	}
	char* input_filename = argv[1];
	std::cout << input_filename << std::endl;
	std::ifstream input_file(input_filename, std::ifstream::in);
	if (input_file.is_open())
	{
		std::string word;
		while (input_file >> word)
		{
			std::cout << word << '\n';
		}
	}
	else
	{
		std::cerr << "Incorrect filename " << input_filename << std::endl;
		exit(1);
	}
    return 0;
}
