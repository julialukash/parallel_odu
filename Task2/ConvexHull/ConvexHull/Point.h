#pragma once
#include <iostream>

class Point
{
private:
	double _x, _y;
	long _position;
	bool _is_convex_hull;
public:

	Point(double x_value = 0.0, double y_value = 0.0, long pos=0)
	{
		_x = x_value;
		_y = y_value;
		_position = pos;
		_is_convex_hull = false;
	}

	double x() const { return _x; }
	double y() const { return _y; }
	long index() const { return _position; }

	double distance (std::shared_ptr<const Point> point) const
	{
		double dx = _x - point->x();
		double dy = _y - point->y();
		return std::sqrt(dx * dx + dy * dy);
	}

	void mark_convex_hull() { _is_convex_hull = true; }
	bool is_convex_hull() const { return _is_convex_hull; }
	
	friend double cross_product(std::shared_ptr<const Point> O, std::shared_ptr<const Point> A, std::shared_ptr<const Point> B)
	{
		return (A->x() - O->x()) * (B->y() - O->y()) - (A->y() - O->y()) * (B->x() - O->x());
	}


	friend bool less_by_x(std::shared_ptr<const Point> point, std::shared_ptr<const Point> other_point)
	{
		return point->x() < other_point->x() || point->x() == other_point->x() && point->y() < other_point->y();
	}

	friend bool less_by_x(std::shared_ptr<Point> point, std::shared_ptr<Point> other_point)
	{
		return point->x() < other_point->x() || point->x() == other_point->x() && point->y() < other_point->y();
	}

	friend std::ostream& operator<<(std::ostream& os, const Point& point)
	{
		os << "(" << point.x() << ", " << point.y() << "), " << point.index() << " position";
		return os;
	}
};