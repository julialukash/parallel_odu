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

	void sort_recursive(std::vector<double>& input, long left, long right)
	{
		if (left < 0 || right > input.size())
		{
			return;
		}

		if (left < right - 1)
		{
			long middle = (left + right) / 2;
			sort_recursive(input, left, middle);
			sort_recursive(input, middle, right);
			merge(input, left, middle, right);
		}
		return;
	}
	
	void merge(std::vector<double>& input, long left, long middle, long right)
	{
		long left_index = 0;
		long right_index = 0;
		std::vector<double> sorted_input_part(right - left);
		while (left + left_index < middle && middle + right_index < right)
		{
			if (input[left + left_index] < input[middle + right_index])
			{
				sorted_input_part[left_index + right_index] = input[left + left_index];
				++left_index;
			}
			else
			{
				sorted_input_part[left_index + right_index] = input[middle + right_index];
				++right_index;
			}
		}

		// copy the rest of array parts
		while (left + left_index < middle)
		{
			sorted_input_part[left_index + right_index] = input[left + left_index];
			++left_index;
		}
		while (middle + right_index < right)
		{
			sorted_input_part[left_index + right_index] = input[middle + right_index];
			++right_index;
		}

		// copy to original array
		for (int i = 0; i < sorted_input_part.size(); ++i)
		{
			input[left + i] = sorted_input_part[i];
		}
		return;
	}
};