#pragma once

#include <vector>

class MergeSorter
{
private:
	bool _is_recursive;
public:
	MergeSorter(bool is_recursive)
	{
		_is_recursive = is_recursive;
	}

	template<typename data>
	void sort_recursive(std::vector<data>& input, long left, long right, bool(*less_comparer)(const data, const data))
	{
		if (left < 0 || right > input.size())
		{
			return;
		}

		if (left < right - 1)
		{
			long middle = (left + right) / 2;
			sort_recursive(input, left, middle, less_comparer);
			sort_recursive(input, middle, right, less_comparer);
			merge(input, left, middle, right, less_comparer);
		}
		return;
	}

	template<typename data>
	void merge(std::vector<data>& input, long left, long middle, long right, bool (*less_comparer)(const data, const data))
	{
		long left_index = 0;
		long right_index = 0;
		std::vector<data> sorted_input_part(right - left);
		while (left + left_index < middle && middle + right_index < right)
		{
			if (less_comparer(input[left + left_index], input[middle + right_index]))
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
		for (size_t i = 0; i < sorted_input_part.size(); ++i)
		{
			input[left + i] = sorted_input_part[i];
		}
		return;
	}
};