#include<vector>
#include<assert.h>

using namespace std;
#pragma region  1. allArrangement

// all possible arrangement
vector<vector<int>>& addNewElemAndGet(vector<vector<int>>& old, int num) {// don't want the 
	assert(old.size() != 0);
	int len = old.size();
	for (int i = 0;i < len;i++) {
		vector<int> origin;
		int sublen = old[i].size();
		for (int j = 0;j <= sublen;j++) {
			if (j == 0) {
				origin = old[i];
				old[i].insert(old[i].begin(), num);
			}
			else {
				origin.insert(origin.begin() + j, num);
				old.push_back(origin);
				origin.erase(origin.begin() + j);
			}
		}
	}
	return old;
}
vector<vector<int>> allArrangement(vector<int> nums) {
	vector<vector<int>> result;
	for (int i = 0;i < nums.size();i++) {
		if (i == 0) {
			result.push_back(vector<int>{nums[i]});
		}
		else {
			addNewElemAndGet(result, nums[i]);
		}
	}
	return result;
}

#pragma endregion



