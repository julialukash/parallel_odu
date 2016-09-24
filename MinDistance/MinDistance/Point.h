#pragma once
#include <iostream>

class Point
{
private:
	double _x, _y;
public:

	Point(double x_value=0.0, double y_value=0.0)
	{
		_x = x_value;
		_y = y_value;
	}

	double x() const { return _x; }
	double y() const { return _y; }

	double distance (const Point* point) const
	{
		double dx = _x - point->x();
		double dy = _y - point->y();
		return std::sqrt(dx * dx + dy * dy);
	}

	friend bool less_by_x(const Point* point, const Point* other_point)
	{
		return point->x() < other_point->x() || point->x() == other_point->x() && point->y() < other_point->y();
	}


	friend bool less_by_y(const Point* point, const Point* other_point)
	{
		return point->y() < other_point->y() || point->y() == other_point->y() && point->x() < other_point->x();
	}

	friend std::ostream& operator<<(std::ostream& os, const Point& point)
	{
		os << "(" << point.x() << ", " << point.y() << ")";
		return os;
	}
};