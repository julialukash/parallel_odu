#pragma once
#include "Point.h"

class ClosestPoints
{
public:
	std::vector<std::shared_ptr<const Point>> first_points;
	std::vector<std::shared_ptr<const Point>> second_points;
	double distance;

	ClosestPoints()
	{
		distance = -1;
	}

	void set_new_distance(std::shared_ptr<const Point> point, std::shared_ptr<const Point> other_point, double new_distance)
	{
		distance = new_distance;
		clear_same_distance_pairs();
		add_same_distance_pair(point, other_point);
	}

	void add_same_distance_pair(std::shared_ptr<const Point> point, std::shared_ptr<const Point> other_point)
	{
		first_points.push_back(point);
		second_points.push_back(other_point);		
	}

	void clear_same_distance_pairs()
	{
		first_points.clear();
		second_points.clear();
	}

	friend std::ostream& operator<<(std::ostream& os, const ClosestPoints& points)
	{
		os << "min distance = " << points.distance << std::endl;
		os << "the closest points are: " << std::endl;
		for (size_t i = 0; i < points.first_points.size(); i++)
		{
			os << *points.first_points[i] << " and " << *points.second_points[i] << std::endl;
		}
		return os;
	}
};