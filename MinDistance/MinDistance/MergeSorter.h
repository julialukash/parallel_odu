#pragma once

#include <vector>
#include "Point.h"

class MergeSorter
{
private:
	bool _is_recursive;
public:
	MergeSorter(bool is_recursive)
	{
		_is_recursive = is_recursive;
	}

	void SortByAscending(std::vector<double>& input, long left, long right)
	{
		return;
	}
	
};