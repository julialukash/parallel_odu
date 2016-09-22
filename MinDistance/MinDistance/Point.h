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

	const double x() { return _x; }
	const double y() { return _y; }

	double distance(Point point)
	{
		double dx = _x - point.x();
		double dy = _y - point.y();
		return std::sqrt(dx * dx + dy * dy);
	}

	friend std::ostream& operator<<(std::ostream& os, Point& point)
	{
		os << "(" << point.x() << ", " << point.y() << ")";
		return os;
	}
};