#include "stdafx.h"
#include "Point.h"
#include "MergeSorter.h"

#include <stdlib.h>   

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
	std::vector<Point> input_points;
	if (input_file.is_open())
	{
		std::string word;
		// read number of dots
		input_file >> word;
		long dots_count = stol(word);
		input_points = std::vector<Point>(dots_count);
		long dot_index = 0;
		while (dot_index < dots_count)
		{
			input_file >> word;
			double x_value = std::atof(word.c_str());
			input_file >> word;
			double y_value = std::atof(word.c_str());
			Point point = Point(x_value, y_value);
			std::cout << point << '\n';
			input_points[dot_index] = point;
			++dot_index;
		}
	}
	else
	{
		std::cerr << "Incorrect filename " << input_filename << std::endl;
		exit(1);
	}
	MergeSorter sorter = MergeSorter(true);
	std::vector<double> xs, ys;
	for each (Point point in input_points)
	{
		xs.push_back(point.x());
		ys.push_back(point.y());
	}
	sorter.SortByAscending(xs, 0, xs.size());
    return 0;
}
