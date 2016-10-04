#include "stdafx.h"
#include "Point.h"
#include "PointsPair.h"
#include "MergeSorter.h"

#include <stdlib.h>   

const int number_of_closest_points = 7;
const double eps = 1e-10;

bool is_equal(double value, double other_value)
{
	return fabs(value - other_value) < eps;
}

void split_by_y(std::vector<std::shared_ptr<const Point>> points, std::shared_ptr<const Point> border_point,
	std::vector<std::shared_ptr<const Point>>& left_points_sorted_by_y,
	std::vector<std::shared_ptr<const Point>>& right_points_sorted_by_y)
{
	long down_points_index = 0, up_points_index = 0;
	for (size_t i = 0; i < points.size(); i++)
	{
		if (less_by_x(points[i], border_point))
		{
			left_points_sorted_by_y[down_points_index++] = points[i];
		}
		else
		{
			right_points_sorted_by_y[up_points_index++] = points[i];
		}
	}
	return;
}

ClosestPoints take_closest_pair(ClosestPoints left_pair, ClosestPoints right_pair)
{
	if (is_equal(left_pair.distance, right_pair.distance))
	{
		for (size_t i = 0; i < right_pair.first_points.size(); i++)
		{
			left_pair.add_same_distance_pair(right_pair.first_points[i], right_pair.second_points[i]);
		}
	}
	return left_pair.distance <= right_pair.distance ? left_pair : right_pair;
}


ClosestPoints find_closest_poinst_in_delta_line(std::vector<std::shared_ptr<const Point>> points_sorted_by_y,
	std::shared_ptr<const Point> border_point, double delta,
	ClosestPoints lef_right_min_distance_pair)
{
	auto min_distance_points = lef_right_min_distance_pair;
	std::vector<std::shared_ptr<const Point>> delta_line_points;
	for each (auto point in points_sorted_by_y)
	{
		if (fabs(point->x() - border_point->x()) <= delta)
		{
			delta_line_points.push_back(point);
		}
	}

	for (size_t n_point = 0; n_point < delta_line_points.size() - 1; n_point++)
	{
		auto point = delta_line_points[n_point];
		auto left_part_point = less_by_x(point, border_point);
		long start_index = n_point + 1;
		long end_index = std::fminl(start_index + number_of_closest_points, delta_line_points.size() - 1);
		for (long i = start_index; i <= end_index; i++)
		{
			auto other_point = delta_line_points[i];
			auto left_part_other_point = less_by_x(other_point, border_point);
			// check points are in different border parts
			if (left_part_point && !left_part_other_point || !left_part_point && left_part_other_point)
			{
				double dist = point->distance(other_point);
				if (dist < min_distance_points.distance)
				{
					min_distance_points.set_new_distance(point, other_point, dist);
				}
				else if (is_equal(dist, min_distance_points.distance))
				{
					min_distance_points.add_same_distance_pair(point, other_point);
				}
			}
		}
	}
	return min_distance_points;
}

ClosestPoints find_closest_points(std::vector<std::shared_ptr<const Point>> points_sorted_by_x,
	std::vector<std::shared_ptr<const Point>> points_sorted_by_y)
{
	ClosestPoints closest_points;
	long n_points = points_sorted_by_x.size();
	if (n_points == 2 || n_points == 3)
	{
		double distance = points_sorted_by_x[0]->distance(points_sorted_by_x[1]);
		closest_points.set_new_distance(points_sorted_by_x[0], points_sorted_by_x[1], distance);
		if (n_points == 3)
		{
			double new_dist_first_last = points_sorted_by_x[0]->distance(points_sorted_by_x[2]);
			double new_dist_second_last = points_sorted_by_x[1]->distance(points_sorted_by_x[2]);
			if (new_dist_first_last < closest_points.distance)
			{
				closest_points.set_new_distance(points_sorted_by_x[0], points_sorted_by_x[2], new_dist_first_last);
			}
			else if (is_equal(new_dist_first_last, closest_points.distance))
			{
				closest_points.add_same_distance_pair(points_sorted_by_x[0], points_sorted_by_x[2]);
			}
			if (new_dist_second_last < closest_points.distance)
			{
				closest_points.set_new_distance(points_sorted_by_x[1], points_sorted_by_x[2], new_dist_second_last);
			}
			else if (is_equal(new_dist_second_last, closest_points.distance))
			{
				closest_points.add_same_distance_pair(points_sorted_by_x[1], points_sorted_by_x[2]);
			}
		}
	}
	else
	{
		long middle_point_index = n_points / 2;
		auto middle_point = points_sorted_by_x[middle_point_index];
		std::vector<std::shared_ptr<const Point>> left_points_sorted_by_x(points_sorted_by_x.begin(), points_sorted_by_x.begin() + middle_point_index);
		std::vector<std::shared_ptr<const Point>> right_points_sorted_by_x(points_sorted_by_x.begin() + middle_point_index, points_sorted_by_x.end());
		std::vector<std::shared_ptr<const Point>> left_points_sorted_by_y(middle_point_index), right_points_sorted_by_y(n_points - middle_point_index);
		split_by_y(points_sorted_by_y, middle_point, left_points_sorted_by_y, right_points_sorted_by_y);
		auto left_closest_points = find_closest_points(left_points_sorted_by_x, left_points_sorted_by_y);
		auto right_closest_points = find_closest_points(right_points_sorted_by_x, right_points_sorted_by_y);
		auto left_right_closest_points = take_closest_pair(left_closest_points, right_closest_points);
		auto delta = left_right_closest_points.distance;
		//merge part
		closest_points = find_closest_poinst_in_delta_line(points_sorted_by_y, middle_point, delta, left_right_closest_points);
	}
	return closest_points;
}



ClosestPoints find_min_distance(std::vector<Point> points)
{
	std::vector<std::shared_ptr<const Point>> points_sorted_by_x(points.size()), points_sorted_by_y(points.size());
	for (size_t i = 0; i < points.size(); i++)
	{
		points_sorted_by_x[i] = std::make_shared<const Point>(points[i]);
		points_sorted_by_y[i] = std::make_shared<const Point>(points[i]);
	}
	auto sorter = MergeSorter(true);
	sorter.sort_recursive(points_sorted_by_x, 0, points_sorted_by_x.size(), less_by_x);
	sorter.sort_recursive(points_sorted_by_y, 0, points_sorted_by_x.size(), less_by_y);
	return find_closest_points(points_sorted_by_x, points_sorted_by_y);
}





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

int main(int argc, char *argv[])
{
	if (argc < 2)
	{
		std::cerr << "Incorrect number of input arguments, please enter filename" << std::endl;
		exit(1);
	}
	auto input_filename = argv[1];
	auto input_points = read_data_from_file(input_filename);
	auto min_dist_points_pair = find_min_distance(input_points);
	std::cout << input_filename << std::endl;
	std::cout << min_dist_points_pair << std::endl;
	return 0;
}
