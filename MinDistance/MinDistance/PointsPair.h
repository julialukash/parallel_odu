#pragma once
#include "Point.h"

class PointsPair
{
public:
	Point point;
	Point other_point;
	double distance;
	PointsPair(Point fist_point=Point(), Point second_point=Point(), double dist = -1)
	{
		point = fist_point;
		other_point = second_point;
		distance = dist;
	}

	friend std::ostream& operator<<(std::ostream& os, const PointsPair& points_pair)
	{
		os << "dist between point " << points_pair.point << " and point " << points_pair.other_point << "\n = " << points_pair.distance;
		return os;
	}
};