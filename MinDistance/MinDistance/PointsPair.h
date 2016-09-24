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
		os << "dist between " << points_pair.point << " and " << points_pair.other_point << " = " << points_pair.distance;
		return os;
	}
};