#include<algorithm>
#include<math.h>
#include<vector>
#include<unordered_map>
#include<unordered_set>
#include<list>
#include<array>
#include<assert.h>
#include<regex>
#include<iostream>
#include"solutions.h"
// The goal:
// 1.recall clrs,
// 2.learn to use stl,
// 3.practice ways of thinking,
// 4.and, possibly, solve the problems.

//principle:
// 1.Index: except some simple operations, all logical index start from 1.
// 2.
using namespace std;
#pragma region 1.Two sum
// 1.brute force: best O(1); worst(n2); average(??);
// 1.use hash: o(n);
vector<int> twoSum(vector<int>& nums, int target) {
	unordered_map<int, pair<int, int>> map;
	for (int i = 0;i < nums.size();i++) {
		if (map[nums[i]].first == 1&& map[target - nums[i]].second>-1) {
			vector<int> tmp;
			tmp.push_back(map[target - nums[i]].second);
			tmp.push_back(i);
			return tmp;
		}
		else {
			map[nums[i]] = pair<int, int>(1, i);
			if(nums[i]!= target - nums[i])
				map[target - nums[i]] = pair<int, int>(1, -1);//the second elem = 0 is for case when 10 = 5+5
		}
	}
	return (vector < int>)0;
}

//remain qustions:
// 1.averge complexity of brute force method ?
// 2.further reduce the time and space
#pragma endregion

#pragma region 2.addTwoNumbers
//1.easy
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
	bool firstFlag = 1;
	ListNode* tail = NULL;
	ListNode* head = tail;
	bool flowed = 0;
	while (l1!=NULL || l2 != NULL) {
		int val =0;
		int tmp1 = 0;
		int tmp2 = 0;
		if (l1 != NULL) {
			tmp1 = l1->val;
			l1 = l1->next;
		}
		if (l2 != NULL) {
			tmp2 = l2->val;
			l2 = l2->next;
		}
		val = tmp1 + tmp2 +flowed;
		flowed = 0;
		if (val > 9) {
			flowed = 1;
			val = val - 10;
		}
		// tail insert
		ListNode* newNode = new ListNode;
		newNode->val = val;
		if (firstFlag) {
			tail = newNode;
			head = tail;
			firstFlag = false;
		}
		else {
			tail->next = newNode;
			tail = newNode;
		}
	}
	if (flowed) {
		ListNode* newNode = new ListNode;
		newNode->val = 1;
		tail->next = newNode;
		tail = newNode;
	}
	return head;
}
#pragma endregion

#pragma region 3.Longest Substring Without Repeating Characters
#pragma region DP with memo and recursion
//1.time o(n2)
// solve using dynamic programming, it didn't pass leetcode submit
//bool have_repeated_elems(string& s, int start, int end) {
//	unordered_set<char> st;
//	for (int i = start;i <= end;i++) {
//		auto rel = st.insert(s[i - 1]);
//		if (!rel.second)
//			return true;
//	}
//	return false;
//}
//int solve(string& s, int start, int end, int** memo) {
//	if (memo[start - 1][end - 1] > -1)
//		return memo[start - 1][end - 1];
//	if (!have_repeated_elems(s, start, end)) {
//		return end - start + 1;
//	};
//	memo[start - 1][end - 1] = 0;// means have repeated value
//	int maxLen = 0;
//	for (int k = start;k < end;k++) {
//		int l1 = 0;
//		if (memo[start - 1][k - 1] > -1)
//			l1 = memo[start - 1][k - 1];
//		else {
//			if (memo[start - 1][k - 1] == 0) {
//				l1 = memo[start - 1][k - 2];
//			}
//			else {
//				l1 = solve(s, start, k, memo);
//			}
//		}
//		int l2 = 0;
//		if (memo[k][end - 1] > -1)
//			l2 = memo[k][end - 1];
//		else {
//			if (memo[k][end - 1] == 0) {
//				l2 = memo[k+1][end - 1];
//			}
//			else {
//				l2 = solve(s, k + 1, end, memo);
//			}
//		}
//		int len = max(l1, l2);
//		if (len > maxLen)
//			maxLen = len;
//	}
//	memo[start - 1][end - 1] = maxLen;
//	return maxLen;
//}
//int lengthOfLongestSubstring(string s) {
//	const int len = s.length();
//	if (len == 0)
//		return 0;
//	int** memo = (int**)malloc(sizeof(int*) * len);
//	for (int i = 0; i < len; i++) {
//		memo[i] = (int*)malloc(sizeof(int) * len);
//		memo[i][i] = 1;
//	}
//	return solve(s, 1, len, memo);
//}
#pragma endregion
#pragma region DP, from bottom to top
//o(n2), it still didn't pass leetcode
//bool have_repeated_elems(string& s, int start, int end) {
//	unordered_set<char> st;
//	for (int i = start;i <= end;i++) {
//		auto rel = st.insert(s[i - 1]);
//		if (!rel.second)
//			return true;
//	}
//	return false;
//}
//int lengthOfLongestSubstring(string s) {
//	const int len = s.length();
//	if (len == 0)
//		return 0;
//	if (len == 1)
//		return 1;
//	int** memo = (int**)malloc(sizeof(int*) * len);
//	for (int i = 0; i < len; i++) {
//		memo[i] = (int*)malloc(sizeof(int) * len);
//		//memo[i][i] = 1;
//	}
//	int maxLen = 1;
//	for (int k = 1;k < len;k++) {
//		for (int i = 1;i <= len - k; i++) {
//			int start = i;int end = i + k;int sublen = 0;
//			if (memo[start - 1][end - 2] == 0) {
//				memo[start - 1][end - 1] == 0;
//			}
//			else if (!have_repeated_elems(s, start, end)) {
//				sublen = end - start + 1;
//				//memo[start - 1][end - 1] = sublen;
//			}
//			if (sublen > maxLen)
//				maxLen = sublen;
//		}
//	}
//	return maxLen;
//}
#pragma endregion
#pragma region greedy
//o(n) base on some oberservation
int lengthOfLongestSubstring(string s) {
	const int len = s.length();
	if (len == 0)
		return 0;
	if (len == 1)
		return 1;
	int maxLen = 1;
	unordered_map<char, int> st;
	int p1 = 1;int p2 = 2;
	int last = p1;
	st.insert({ s[p1 - 1] ,p1 });
	for (int p2 = 2;p2 <= len;p2++) {
		auto rel = st.insert({ s[p2 - 1] ,p2 });
		if (!rel.second) {

			last = p1;
			int tmp = st[s[p2 - 1]];
			p1 = tmp + 1;
			for (int i = last;i <= tmp;i++)
				st.erase(s[i - 1]);
			st[s[p2 - 1]] = p2;
		}
		if (p2 - p1 + 1 > maxLen) {
			maxLen = p2 - p1 + 1;
		}
	}
	return maxLen;
}
#pragma endregion
#pragma endregion

#pragma region 5.Longest Palindromic Substring
//#pragma region DP,bad
////o(n3) did not pass leetcode
//bool is_palindromic(string s, int start, int end) {
//	if (s.length() == 0)
//		return true;
//	if (start == end)
//		return true;
//	int len = end - start + 1;
//	for (int i = start;i <= (start + end) / 2;i++) {
//		if (s[i - 1] != s[end - (i - start) - 1]) {
//			return false;
//		}
//	}
//	return true;
//}
//string longestPalindrome(string s) {
//	if (is_palindromic(s, 1, s.length()))
//		return s;
//	int len = s.length();
//	string maxSub = s.substr(0, 1);
//	int maxLen = 1;
//	int p1 = 1;int p2 = 1;
//	for (int k = 1;k <= len;k++) {
//		for (p1 = 1;p1 <= len - k;p1++) {
//			p2 = p1 + k;
//			if (s[p1 - 1] == s[p2 - 1]) {
//				if (is_palindromic(s, p1, p2)) {
//					if (p2 - p1 + 1 > maxLen) {
//						maxLen = p2 - p1 + 1;
//						maxSub.clear();
//						maxSub = s.substr(p1 - 1, maxLen);//left close rigth open
//					}
//				}
//			}
//		}
//	}
//	return maxSub;
//}
//#pragma endregion
#pragma region greedy
pair<int,int> expand(string s,int center) {
	int left = center;int right = center;
	char c = s[center];
	while ((left>0&&s[left - 1] == c)|| (right<=s.length() &&s[right - 1] == c)) {
		if ((left > 0 && s[left - 1] == c))
			left--;
		if ((right <= s.length() && s[right - 1] == c))
			right++;
	}
	bool allsameflag = true;
	int len = s.length();
	int count = 0;
	while (left > 0 && right <= len) {
		if (s[left-1] != s[right-1]) {
			count++;
			break;
		}
		left--;right++;
	}
	return pair<int, int>(left+1,right-1);
}
string longestPalindrome(string s) {
	int len = s.length();
	if (len == 0 || len == 1)
		return s;
	int maxLen = 0;string maxSub = s.substr(0,1);
	for (int i = 1;i <= len;i++) {
		pair<int, int>range = expand(s, i);
		int left = range.first;int right = range.second;
		if (right - left + 1 > maxLen) {
			maxLen = right - left + 1;
			maxSub.clear();
			maxSub = s.substr(left - 1, maxLen);
		}
	}
	return maxSub;
}
#pragma endregion

#pragma endregion

#pragma region 6.ZigZag Conversion
// the test case is corrects, but did not pass leetcode submit for unknown reason, we can also using mapping,which is less time comsuming
void fillMatrix(string s, char** m, int ithBatch,int numColsPerBatch,int numRows, int numElemPerBatch) {
	int startCol = (ithBatch - 1) * numColsPerBatch;
	int endCol   = (ithBatch ) *numColsPerBatch;
	int eleIdx = (ithBatch-1)*numElemPerBatch;
	for (int j = startCol;j < endCol;j++) {//which column
		for (int i = 0;i < numRows;i++) {
			if (j == startCol|| i == numRows-1 - (j - startCol)) {
				if(eleIdx<s.length())
					m[i][j] = s[eleIdx];
				else {
					m[i][j] = ' ';
				}
				eleIdx++;
			}
			else {
				m[i][j] = ' ';
			}
		}
	}
}
string readHorizontal(char**m,int numRows,int numCols) {
	string s;
	for (int i = 0;i < numRows;i++) {
		for (int j = 0;j < numCols;j++) {
			if(m[i][j]!=' ')
				s.push_back(m[i][j]);
		}
	}
	return s;
}
string convert(string s, int numRows) {
	if (s.length() <= numRows||numRows==1) {
		return s;
	}
	int numColsPerBatch = numRows - 1;
	int numElemPerBatch = numRows + numColsPerBatch - 1;
	int batches = ceil(float(s.length() )/ numElemPerBatch);
	int numCols = batches * numColsPerBatch;

	char** m = (char**)malloc(sizeof(char*)* numRows);
	for (int i = 0;i < numRows;i++) {
		m[i] = (char*)malloc(sizeof(char)* numCols);
		memset(m[i],' ', sizeof(char) * numCols);
	}
	int len = s.length();int i = 0; int ithBatch = 1;
	for (;i < len;i += numElemPerBatch,ithBatch++) {
		fillMatrix(s, m, ithBatch, numColsPerBatch, numRows, numElemPerBatch);
	}
	string result = readHorizontal(m,numRows,numCols);
	return result;
}
#pragma endregion

#pragma region 7. Reverse Integer
int reverse(int x) {
	int result = 0;
	int bits = 0;
	while (x!=0) {
		int res = x % 10;
		x = x / 10;
		if (result > INT_MAX / 10 ||(result == INT_MAX / 10&& res > INT_MAX % 10))
			return 0;
		if (result < INT_MIN / 10 || (result == INT_MIN / 10 && res < INT_MIN % 10))
			return 0;
		result = result * 10 + res;
	}
	return result;
}
#pragma endregion

#pragma region 8.String to Integer (atoi)
// should not expect to get it right the first time
int myAtoi(string s) {
	int result = 0;
	int i = 0;bool neg = false;
	for (;i < s.length();i++) {
		if ('0' <= s[i] && s[i] <= '9')
			break;
		if (s[i] == ' ') {
			continue;
		}
		else if (s[i] == '-'|| s[i] == '+') {
			if (s[i] == '-')neg = true;
			if (s[i] == '+')neg = false;
			if (i+1 < s.length()&&'0' <= s[i+1] && s[i + 1] <= '9')
				i++;
				break;
			return 0;
		}
		else if (('0' > s[i] || s[i] > '9') && (s[i] != '-' || s[i] != '+') && result == 0) {
			return 0;
		}
	}
	while ('0' <= s[i] && s[i] <= '9' &&i<s.length()) {
		int res = s[i] - '0';
		if (neg==false &&(result > INT_MAX / 10 || (result == INT_MAX / 10 && res > INT_MAX % 10)))
			return INT_MAX;
		if (neg == true && (result > -(INT_MIN / 10) || (result == -(INT_MIN / 10) && res >= -(INT_MIN % 10))))
			return INT_MIN;
		result = result * 10 + res;
		i++;
	}
	result = neg ? -result : result;
	return result;
}
#pragma endregion

#pragma region 9. Palindrome Number
bool isPalindrome(int x) {
	if (x < 0) {
		return false;
	}
	if (x < 10) {
		return true;
	}
	int len = 0;
	int x2 = x;
	while (x2 != 0) {
		x2 = x2 / 10;
		len++;
	}
	vector<int> memo;
	int count = 0;
	while (count<len/2) {
		memo.push_back(x % 10);
		x = x / 10;
		count++;
	}
	if (len % 2 == 1) {
		x = x / 10;// erase middle bit
	}
	while (x != 0) {
		if ((x % 10) != memo.back()) {
			return false;
		}
		memo.pop_back();
		x = x / 10;
	}
	return true;
}
#pragma endregion


#pragma region 10. Regular Expression Matching
#pragma region into blocks
// 1. se
//int matchStar(string s,int pos) {
//	//must s[pos] must exist
//	int i = 0;
//
//	for (i = pos;i < s.length();i++) {
//		if (s[pos] != s[i]) {
//			break;
//		}
//	}
//	return i-pos;
//}
//int matchDotStar(string s,int pos,char stop) {
//	if (pos >= s.length())
//		return 0;
//	int i = 0;
//	for (i = pos;i < s.length();i++) {
//		if (stop != s[i])
//			break;
//	}
//	return i - pos + 1;
//}
//bool isMatch(string s, string p) {
//	if (s.length() == 0 || p.length() == 0)
//		return false;
//	int ppos = 0;
//	int spos = 0;
//	while (spos<=s.length()) {
//		if (ppos == p.length() && spos < s.length())
//			return false;
//		else if (p[ppos] == '.') {
//			if (ppos + 1 < p.length() && p[ppos + 1] == '*') {
//				if (ppos + 2 < p.length()) {
//					int len = matchDotStar(s, spos, p[ppos + 2]);
//					spos += len;ppos += 2;
//				}
//				else {
//					return true;
//				}
//			}
//			else {
//				spos++;ppos++;
//			}
//		}
//		else if (s[spos] == p[ppos]) {
//			if (ppos + 1 < p.length() && p[ppos + 1] == '*') {
//				int len = 0;
//				if (ppos + 2 < p.length()&& p[ppos + 2] == s[spos]) {
//					int tmp = ppos + 2;
//					while (tmp<p.length()) {
//						if (p[tmp] != s[spos])
//							break;
//						tmp++;
//					}
//					tmp--;int len = tmp - ppos - 2+1;
//					for (int i = 0;i < len;i++) {
//						spos += i;
//						if (s[spos] != s[spos])
//							return false;
//					}
//					ppos += 2;
//				}
//				else {
//					len = matchStar(s, spos);
//					spos += len;ppos += 2;
//				}
//			}
//			else {
//				spos++;ppos++;
//			}
//		}
//		else if (s[spos] != p[ppos]) {
//			if (ppos + 1 < p.length() && p[ppos + 1] == '*') {
//				ppos+=2;
//			}
//			else {
//				return false;
//			}
//		}
//	}
//	return true;
//}
// the above code didn't pass s="a*a", p="aaa", I give up, I think I don't fully understand what regular expression is, so I can't solve it with dp and other alogrithms
#pragma endregion

#pragma region stl
bool isMatch(string s, string p) {
	std::regex txt_regex(p);
	return regex_match(s, txt_regex);
}
#pragma endregion
#pragma endregion

#pragma region 11.Container With Most Water
#pragma region o(n2) 
////1. didn't pass, may need nlogn
//int maxArea(vector<int>& height) {
//	if (height.size() < 2) {
//		return 0;
//	}
//	int len = height.size();
//	int maxArea = 0;
//	for (int i = 0;i < len;i++) {
//		for (int j = i + 1;j < len;j++) {
//			int area = min(height[i], height[j]) * (j - i);
//			if (area > maxArea) {
//				maxArea = area;
//			}
//		}
//	}
//	return maxArea;
//}
#pragma endregion

int maxArea(vector<int>& height) {
	if (height.size() < 2)
		return 0;

	vector<pair<int, int>> p;
	for (int i = 0;i < height.size();i++) {
		p.push_back(pair<int, int>(height[i],i));
	}
	struct {
		bool operator()(pair<int, int> a, pair<int, int> b) const {
			return a.first > b.first;
		}
	} customGreater;
	sort(p.begin(), p.end(), customGreater);

	int maxArea = min(p[0].first,p[1].first)*(abs(p[0].second - p[1].second));
	pair<int, int> maxMin(p[0].second, p[1].second);int maxInterval = abs(p[0].second - p[1].second);
	for (int i = 2;i < p.size();i++) {
		int min1 = min(min(maxMin.first, maxMin.second),p[i].second);
		int max1 = max(max(maxMin.first, maxMin.second),p[i].second);
		maxMin.first = min1;
		maxMin.second = max1;
		int in = max(abs(p[i].second - maxMin.first), abs(p[i].second - maxMin.second));
		if (in > maxInterval) {
			int area = p[i].first*in;
			if (area>maxArea) {
				maxArea = area;
				maxInterval = in;
			}
		}
	}
	return maxArea;
}
#pragma endregion

#pragma region 14. Longest Common Prefix

string longestCommonPrefix(vector<string>& strs) {
	string s("");
	if (strs.size() == 0)
		return "";
	if (strs.size() == 1)
		return strs[0];
	while (1) {
		for (int i = 0;i < strs.size();i++) {
			if (strs[i].length() == 0)
				return "";
			if (s.length() > strs[i].length()) {
				s.pop_back();
				return s;
			}
			else {
				if (i == 0) {
					char c = strs[i][s.length()];
					s.push_back(c);
				}
				else if(strs[i][s.length()-1] != s[s.length() - 1]) {
					s.pop_back();
					return s;
				}
			}	
		}
	}
}
#pragma endregion

#pragma region 15. 3Sum

#pragma region o(n2)
// pass 315 / 318 tests, Time limit exceeded
// basic idea: find result with three zeros, 1 zeros, and no zeros;
typedef pair<pair<int, int>, pair<int, int>> myType;
struct myHash
{
	size_t operator()(const myType& r1) const {// have effect on insertion!,avoid same keys.
		//return hash<int>{}(r1.first.first* r1.second.first);// this function works only in two sum cases
		return hash<int>{}(hash<int>()(r1.first.first) ^ hash<int>()(r1.second.first));// the above line may cause overflow
	}
};
struct myEqual
{
	bool operator()(const myType& rc1, const myType& rc2) const noexcept {// don't have effect on insertion?
		return (rc1.first.first == rc2.first.first
			&& rc1.second.first == rc2.second.first)
			|| (rc1.first.first == rc2.second.first
				&& rc1.second.first == rc2.first.first);
	}
};
vector<vector<int>> setToVector(unordered_set<myType, myHash, myEqual> set) {
	vector<vector<int>> result;
	for (unordered_set<myType>::iterator i = set.begin();i != set.end();i++) {
		vector<int> tmp; tmp.push_back((*i).first.second);tmp.push_back((*i).second.second);
		result.push_back(tmp);
	}
	return result;
}
vector<vector<int>> twoSum2(vector<int>nums, int target) {
	// different from towSum with two distinctions:1.result is not unique;2.nums is sorted from small to big
	unordered_set<myType, myHash, myEqual> result;//two pair, each pair is(value, indx) group that satisfiy target
	if (nums.size() < 2)
		return setToVector(result);
	unordered_map<int, pair<int, int>> map;
	for (int i = 0;i < nums.size();i++) {
		if (map[nums[i]].first == 1 && map[target - nums[i]].second > -1) {
			myType tmp1;
			tmp1.first.first = nums[i];tmp1.first.second = i;
			tmp1.second.first = target - nums[i];tmp1.second.second = map[target - nums[i]].second;
			result.insert(tmp1);
		}
		else {
			map[nums[i]] = pair<int, int>(1, i);
			if (nums[i] != target - nums[i])
				map[target - nums[i]] = pair<int, int>(1, -1);//the second elem = 0 is for case when 10 = 5+5
		}
	}
	return setToVector(result);
}
int findNum(vector<int>& nums, vector<int>::iterator first, vector<int>::iterator second, int num) {
	vector<int>::iterator result = find(first, second, num);
	if (result != nums.end()) {
		return result - nums.begin();
	}
	return -1;
}
vector<vector<int>> threeSum(vector<int>& nums) {
	vector<vector<int>> result;
	if (nums.size() < 3)
		return vector<vector<int>>();
	if (nums.size() == 3 && (nums[0] + nums[1] + nums[2] == 0)) {
		result.push_back(nums);
		return result;
	}
	sort(nums.begin(), nums.end());
	int idx = findNum(nums, nums.begin(), nums.end(), 0);
	// if nums have 0,get Modsat nummer
	if (idx > -1) {
		vector<int> sub;
		if (idx + 2 < nums.size() && nums[idx] == 0 && nums[idx + 1] == 0 && nums[idx + 2] == 0) {// all three zeros, after processing, only left one zero
			vector<int> subNum;subNum.push_back(0);subNum.push_back(0);subNum.push_back(0);
			result.push_back(subNum);
		}
		for (int i = idx + 1;i < nums.size();i++) {//only left one zero
			if (nums[i] == 0) { nums.erase(nums.begin() + i);i--; }
			else {
				break;
			}
		}
		for (int i = 0;i < idx;i++) {//only have one zero
			sub.push_back(0);
			sub.push_back(nums[i]);
			int idx2 = -1;
			if (i == 0 || (i > 0 && nums[i - 1] != nums[i]))
				idx2 = findNum(nums, nums.begin() + idx, nums.end(), -nums[i]);
			if (idx2 > -1) {
				sub.push_back(-nums[i]);
				result.push_back(sub);
			}
			sub.clear();
		}
		nums.erase(nums.begin() + idx);
	}
	// one nums don't have zeros, we can convert to 2 sum problem
	int splitIdx;//point to the last negative value
	for (splitIdx = 0;splitIdx < nums.size();splitIdx++) {
		if (splitIdx + 1 < nums.size() && nums[splitIdx + 1] > 0) {
			break;
		}
	}
	if (splitIdx >= nums.size()) return result;
	for (int i = 0;i < nums.size();i++) {
		if (i <= splitIdx) {
			vector<int> subNum(nums.begin() + splitIdx + 1, nums.end());
			vector<vector<int>> idx2 = { { -1} };
			if (i == 0 || (i > 0 && nums[i - 1] != nums[i]))
				idx2 = twoSum2(subNum, -nums[i]);
			vector<int> sub;
			for (int j = 0;j < idx2.size();j++) {
				if (idx2[j].size() == 2) {
					sub.push_back(nums[i]);sub.push_back(nums[idx2[j][0] + splitIdx + 1]);sub.push_back(nums[idx2[j][1] + splitIdx + 1]);
					result.push_back(sub);
					sub.clear();
				}
			}
		}
		if (i > splitIdx) {
			vector<int> subNum(nums.begin(), nums.begin() + splitIdx + 1);
			vector<vector<int>> idx2 = { { -1} };
			if (i == 0 || (i > 0 && nums[i - 1] != nums[i]))
				idx2 = twoSum2(subNum, -nums[i]);
			vector<int> sub;
			for (int j = 0;j < idx2.size();j++) {
				if (idx2[j].size() == 2) {
					sub.push_back(nums[i]);sub.push_back(nums[idx2[j][0]]);sub.push_back(nums[idx2[j][1]]);
					result.push_back(sub);
					sub.clear();
				}
			}
		}
	}
	return result;
}
#pragma endregion


#pragma endregion





