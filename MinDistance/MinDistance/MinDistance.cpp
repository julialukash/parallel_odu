#include "stdafx.h"
#include "Point.h"
#include "PointsPair.h"
#include "MergeSorter.h"

#include <stdlib.h>   


std::vector<Point> read_data_from_file(std::string filename);

PointsPair find_min_distance(std::vector<Point> points, MergeSorter* sorter);


int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Incorrect number of input arguments, please enter filename"<< std::endl;
		exit(1);
	}
	char* input_filename = argv[1];
	std::cout << input_filename << std::endl;
	std::vector<Point> input_points = read_data_from_file(input_filename);
	MergeSorter sorter = MergeSorter(true);
	PointsPair min_dist_points_pair = find_min_distance(input_points, &sorter);
	std::cout << min_dist_points_pair << std::endl;
	return 0;
}

PointsPair find_min_distance(std::vector<Point> points, MergeSorter* sorter)
{
	std::vector<Point*> points_sorted_by_x;
	return PointsPair(points[0], points[1], -2);
}


std::vector<Point> read_data_from_file(std::string filename)
{
	std::ifstream input_file(filename, std::ifstream::in);
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
		std::cerr << "Incorrect filename " << filename << std::endl;
		exit(1);
	}	
	return input_points;
}