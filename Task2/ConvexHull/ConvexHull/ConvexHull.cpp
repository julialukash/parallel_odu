#include "stdafx.h"
#include "Point.h"
#include "MergeSorter.h"

#include <stdlib.h>   
#include <time.h>

std::vector<Point> read_data_from_file(std::string filename)
{
	std::ifstream input_file(filename, std::ifstream::in);
	std::vector<Point> input_points;
	if (input_file.is_open())
	{
		std::string word;
		//read number of dots
		input_file >> word;
		long dots_count = stol(word);
		if (dots_count <= 1)
		{
			std::cerr << "Incorrect number of dots = " << dots_count << std::endl;
			exit(1);
		}
		input_points = std::vector<Point>(dots_count);
		long dot_index = 0;
		while (dot_index < dots_count)
		{
			input_file >> word;
			double x_value = std::atof(word.c_str());
			input_file >> word;
			double y_value = std::atof(word.c_str());
			Point point = Point(x_value, y_value, dot_index);
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

std::vector<std::shared_ptr<const Point>> delete_from_vector_by_flags(std::vector<std::shared_ptr<const Point>> points, std::vector<bool> flags)
{
	std::vector<std::shared_ptr<const Point>> left_points;
	for (auto i = 0; i < points.size(); ++i)
	{
		if (!flags[i])
			left_points.push_back(points[i]);
	}
	return left_points;
}

std::vector<std::shared_ptr<Point>> delete_convex_hull_from_vector(std::vector<std::shared_ptr<Point>> points)
{	
	std::vector<std::shared_ptr<Point>> left_points;
	for (auto i = 0; i < points.size(); ++i)
	{		
		if (!points[i]->is_convex_hull())
			left_points.push_back(points[i]);
	}
	return left_points;
}

std::vector<std::shared_ptr<Point>> get_convex_hull(std::vector<std::shared_ptr<Point>> points_sorted_by_x)
{
	auto points_count = points_sorted_by_x.size();
	if (points_count == 1)
	{
		points_sorted_by_x[0]->mark_convex_hull();
		return points_sorted_by_x;
	}
	std::vector<std::shared_ptr<Point>> convex_hull(2 * points_count);

	int k = 0;
	// Build lower hull
	for (auto i = 0; i < points_count; ++i)
	{
		while (k >= 2 && cross_product(convex_hull[k - 2], convex_hull[k - 1], points_sorted_by_x[i]) < 0)
			k--;
		convex_hull[k++] = points_sorted_by_x[i];
	}

	// Build upper hull
	for (int i = points_count - 2, t = k + 1; i >= 0; i--)
	{
		while (k >= t && cross_product(convex_hull[k - 2], convex_hull[k - 1], points_sorted_by_x[i]) < 0)
			k--;
		convex_hull[k++] = points_sorted_by_x[i];
	}
	convex_hull.resize(k - 1);
	for each (auto point in convex_hull)
	{
		point->mark_convex_hull();
	}
	return convex_hull;
}


std::vector<int> get_deep_function(std::vector<Point> input_points)
{
	std::vector<int> deep_function_values;

	// Sort points lexicographically
	std::vector<std::shared_ptr<Point>> points_sorted_by_x(input_points.size());
	for (size_t i = 0; i < input_points.size(); i++)
	{
		points_sorted_by_x[i] = std::make_shared<Point>(input_points[i]);
	}
	auto sorter = MergeSorter(true);
	sorter.sort_recursive(points_sorted_by_x, 0, points_sorted_by_x.size(), less_by_x);

	int deep = 1;
	while (points_sorted_by_x.size() != 0)
	{
		auto convex_hull = get_convex_hull(points_sorted_by_x);
		deep_function_values.push_back(convex_hull.size());
		++deep;

		// remove convex hull from points
		points_sorted_by_x = delete_convex_hull_from_vector(points_sorted_by_x);
	}
	return deep_function_values;
}

void print_deep_function(std::vector<int> values)
{
	for (auto i = 0; i < values.size(); ++i)
	{
		std::cout << i + 1 << "\t" << values[i] << std::endl;
	}
}

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Incorrect number of input arguments, please enter filename" << std::endl;
		exit(1);
	}
	auto input_filename = argv[1];
	auto input_points = read_data_from_file(input_filename);
	auto begin = clock();
	auto deep_function_values = get_deep_function(input_points);
	auto end = clock();
	auto time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	std::cout << input_filename << std::endl;
	print_deep_function(deep_function_values);
	std::cout << "time spent: " << time_spent << std::endl;
    return 0;
}

