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
#include< queue >
#include<map>
#include<set>
#include<stack>
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


#pragma region 17. Letter Combinations of a Phone Number
vector<string> newArrangement(vector<string>& origin, vector<string> target) {
	if (origin.size() == 0) {
		origin = target;return origin;
	}
	int len = origin.size();
	for (int i = 0;i < len;i++) {
		string tmp;
		for (int j = 0;j < target.size();j++) {
			if (j == 0) {
				tmp = (origin[i]);
				origin[i].append(target[j]);
			}
			else {
				tmp.append(target[j]);
				origin.push_back(tmp);
				tmp.pop_back();
			};
		}
	}
	return origin;
}
vector<string> letterCombinations(string digits) {
	unordered_map<string,vector<string>> map;
	map["2"] = {"a","b","c"};
	map["3"] = {"d","e","f"};
	map["4"] = {"g","h","i"};
	map["5"] = {"j","k","l"};
	map["6"] = {"m","n","o"};
	map["7"] = {"p","q","r","s"};
	map["8"] = {"t","u","v"};
	map["9"] = {"w","x","y","z"};
	vector<string> result;
	for (auto i = 0;i <digits.length();i++) {
		newArrangement(result,map[digits.substr(i,1)]);
	}
	return result;
}
#pragma endregion

#pragma region 19. Remove Nth Node From End of List

ListNode* removeNthFromEnd(ListNode* head, int n) {
	if (head->next == NULL)
		return NULL;
	ListNode* p1 = head;
	ListNode* p2 = head;
	ListNode* pre = NULL;
	for (int i = 1;i < n;i++) {
		p1 = p1->next;
	}
	while (p1->next != NULL) {
		pre = p2;
		p2 = p2->next;
		p1 = p1->next;
	}
	if (p2 == head) {
		head = p2->next;
	}
	else {
		pre->next = p2->next;
	}
	delete(p2);
	return head;
}

#pragma endregion

#pragma region 20. Valid Parentheses
bool isValid(string s) {
	if (s.length() % 2 == 1)
		return false;
	unordered_map<string, string> left;left["{"] = "}";left["("] = ")";left["["] = "]";
	unordered_map<string, string> right;
	vector<string> stack;
	for (int i = 0;i < s.length();i++) {
		if (s.substr(i, 1) == "{") { stack.push_back("{"); continue; }
		else if (s.substr(i, 1) == "(") { stack.push_back("("); continue; }
		else if (s.substr(i, 1) == "[") { stack.push_back("["); continue; };
		
		if (stack.empty() || left[stack.back()] != s.substr(i, 1))
			return false;
		else {
			stack.pop_back();
		}
	}
	if (stack.size() != 0)
		return false;
	return true;
}
#pragma endregion

#pragma region 21. Merge Two Sorted Lists
// passed leetcode,but can not run on my own pc
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	if (l1 == NULL)
		return l2;
	if (l2 == NULL)
		return l1;
	ListNode* head;
	ListNode* p1;ListNode* p2;// p1 point to the return list 
	if (l1->val <= l2->val) {
		head = l1;p1 = l1;p2 = l2;
	}
	else {
		head = l2;p1 = l2;p2 = l1;
	}
	ListNode* pre= p1;
	while (p2 != NULL) {
		if (p1->val < p2->val) {
			pre = p1;
			p1 = p1->next;
			if (p1 != NULL) {
				continue;
			}
			else {
				pre->next = p2;
				break;
			}
		}
		else {
			ListNode* tmp = p2->next;
			p2->next = pre->next;
			pre->next = p2;
			pre = p2;
			p2 = tmp;
		}
	}
	return head;
}
#pragma endregion

#pragma region 23. Merge k Sorted Lists

#pragma region find the minimum of each
//O(nk) k = list.size()
// we can use divide and conquer to reduce O(nk) to O(nlogk)
ListNode* mergeKLists(vector<ListNode*>& lists) {
	ListNode* head = NULL;
	ListNode* p = NULL;
	while (lists.size() != 0) {
		ListNode** minNode = NULL;
		for (int i = 0;i < lists.size();i++) {
			if (lists[i] == NULL) {
				lists.erase(lists.begin() + i);
				i--;
			}
			else {
				minNode = &lists[i];break;
			}
		}
		for (int i = 0;i < lists.size();i++) {
			ListNode* sub = lists[i];
			if (sub == NULL) {
				lists.erase(lists.begin() + i);
				i--;continue;
			}
			if (sub->val < (*minNode)->val) {
				minNode = &lists[i];
			}
		}
		if (head == NULL) {
			if (minNode == NULL)
				break;
			head = *minNode;
			(*minNode) = (*minNode)->next;// detach this node from lists

			p = head;p->next = NULL;continue;
		}
		if (minNode == NULL)
			break;
		else {
			p->next = (*minNode);// add to result
			(*minNode) = (*minNode)->next;// detach this node from lists

			p = p->next;
			p->next = NULL;
		}
	}
	return head;
}
#pragma endregion

#pragma region Devide and conquer

#pragma endregion

#pragma endregion

#pragma region 24. Swap Nodes in Pairs
bool swapKNodes(ListNode** subhead, ListNode* preNode, const int K) {
	if (*subhead == NULL)
		return false;
	ListNode* l1= *subhead;
	int count=0;
	stack<ListNode*> s;
	//ListNode* pre = preNode;
	while (count!=K) {
		if (count == 0) {
			//pre = preNode;
			s.push(l1);
			count++;
		}
		else {
			//pre = l1;
			l1 = l1->next;
			if (l1 == NULL)// the length of the list is less than K,so do nothing
				return false;
			s.push(l1);
			count++;
		}
	}
	ListNode* postNode = l1->next;
	ListNode* tail = s.top(); s.pop();
	ListNode* subHead = tail;
	while (s.size() != 0) {
		tail->next = s.top();s.pop();
		tail = tail->next;
	}tail->next = NULL;
	if (preNode == NULL) {
		(*subhead) = subHead;
	}
	else {
		preNode->next = subHead;
	}
	if (postNode != NULL)
		tail->next = postNode;
	return true;
}
ListNode* swapPairs(ListNode* head) {
	if (head == NULL)
		return NULL;
	ListNode* p=head;
	ListNode* pre=NULL;
	int result = true;
	while (p!=NULL && result==true) {
		if (pre == NULL) {
			result = swapKNodes(&head, pre, 2);
		}
		else {
			result = swapKNodes(&p, pre, 2);
		}
		if (p->next != NULL) {
			pre = p;
			p = p->next;
		}
		else {
			return head;
		}
	}
	return head;
}
#pragma endregion

#pragma region 25. Reverse Nodes in k-Group
ListNode* reverseKGroup(ListNode* head, int k) {
	if (head == NULL)
		return NULL;
	ListNode* p = head;
	ListNode* pre = NULL;
	int result = true;
	while (p != NULL && result == true) {
		if (pre == NULL) {
			result = swapKNodes(&head, pre, k);
		}
		else {
			result = swapKNodes(&p, pre, k);
		}
		if (p->next != NULL) {
			pre = p;
			p = p->next;
		}
		else {
			return head;
		}
	}
	return head;
}
#pragma endregion

#pragma region 26. Remove Duplicates from Sorted Array
int removeDuplicates(vector<int>& nums) {
	for (auto it = nums.begin();it != nums.end();) {
		if (it + 1 != nums.end() && *(it + 1) == *(it)) {
			nums.erase(it);
		}
		else {
			it++;
		}
	}
	return nums.size();
}
#pragma endregion

#pragma region 27. Remove Element
int removeElement(vector<int>& nums, int val) {
	for (auto it = nums.begin();it != nums.end();) {
		if (*(it) == val) {
			nums.erase(it);
		}
		else {
			it++;
		}
	}
	return nums.size();
}
#pragma endregion

#pragma region 28. Implement strStr()
int strStr(string haystack, string needle) {
	int len = needle.length();
	for (auto it = haystack.begin();it <= (haystack.end() - len);) {
		if (needle == haystack.substr(it - haystack.begin(), len)) {
			return it - haystack.begin();
		}
		else {
			it++;
		}
	}
	return -1;
}
#pragma endregion

#pragma region 30. Substring with Concatenation of All Words
// exceed leetcode runing time
bool subStrEqual(string s, unordered_multiset<string> ms) {
	int len = ms.begin()->length();
	int check_len = ms.size() * len;
	for (int i = 0;i < check_len;i+=len) {
		unordered_multiset<string>::iterator res = ms.find(s.substr(i, len));
		if (res == ms.end()) {
			return false;
		}
		else {
			ms.erase(res);
		}
	}
	if(ms.size()==0)
		return true;
	else {
		return false;
	}
}
vector<int> findSubstring(string s, vector<string>& words) {
	if (s.length() < words[0].length() * words.size())
		return {};
	
	unordered_multiset<string> ms;
	for (int i = 0;i < words.size();i++) {
		ms.insert(words[i]);
	}
	vector<int> result;
	for (int i = 0;i < s.length();i++) {
		if (subStrEqual(s.substr(i, s.length() - i), ms)) {
			result.push_back(i); 
		}
		else {
		}
	}
	return result;
}
#pragma endregion

#pragma region 31. Next Permutation
inline void swapIntVector(vector<int>::iterator it1, vector<int>::iterator it2) {
	int tmp = *(it1);*(it1) = *(it2);*(it2) = tmp;
}
void nextPermutation(vector<int>& nums) {
	if (nums.size() <= 1)
		return;
	auto it = nums.end() - 1;
	vector<int>::iterator firstBigger;
	while ((it) != (nums.begin()) && *(it - 1) >= *(it)) {
		it--;
	}
	vector<int>::iterator ith;		vector<int>::iterator itt;
	if (it != nums.begin()) {
		int target = *(it - 1);
		auto it2 = nums.end() - 1;
		vector<int>::iterator firstBigger = nums.end();
		while ((it2) != it - 1) {
			if (firstBigger == nums.end()) {
				if (*it2 > target) {
					firstBigger = it2;
				}
			}
			else {
				if (*it2 > target && *it2 < *firstBigger) {
					firstBigger = it2;
				}
			}
			it2--;
		}
		swapIntVector(firstBigger, it - 1);
		ith = it;
	}
	else {
		ith = nums.begin();
	}
	itt = nums.end() - 1;
	while (ith < itt) {
		swapIntVector(ith, itt);
		ith++;itt--;
	}
}
#pragma endregion

#pragma region 32. Longest Valid Parentheses
int longestValidParentheses(string s) {
	stack<string> st;
	int max = 0;
	int subLen = 0;auto it = s.begin();
	while(it != s.end()) {
		if (st.size() == 0) { subLen = 0; it++; }
		if (s.substr(it - s.begin(), 1) == "(") {
			if (subLen==0) {
				st.push("(");it++;continue;
			}
			else {
				while (!st.empty()) {
					st.pop();
				}
			}
		}
		else {
			 if( st.top() == "(")
			{
				 subLen += 2;st.pop();it++;
				 while (s.substr(it-s.begin(),1) != "(") {
					 subLen += 2;st.pop();it++;
				 }
				 continue;
			}
			if (subLen > max) {
				max = subLen;
			}
		}
	}
	return max;
}
#pragma endregion

#pragma region 33. Search in Rotated Sorted Array
int search(vector<int>& nums, int target) {
	if (nums.size() == 0)
		return -1;
	if (nums.size() == 1) {
		if (nums[0] == target)
			return 0;
		else {
			return -1;
		}
	}
	if(nums[0]<nums[nums.size()-1])
	{
		auto idx = lower_bound(nums.begin(), nums.end(), target);
		if (idx != nums.end()&&*idx==target)
			return idx - nums.begin();
		else {
			return -1;
		}
	}
	if ( nums[0] < target) {
		auto first = nums.begin();
		auto sec = nums.end() - 1;
		while (first>sec) {
			if (first - sec <= 1) {
				for (auto it = first;it <= sec;it++) {
					if (*it == target) {
						return it - nums.begin();
					}
				}
				return -1;
			}
			auto half = (sec - first+1)/2;
			auto mid = first + half;
			if (*mid == target)
				return mid - nums.begin();
			if (*mid > target) {
				auto idx = lower_bound(first, mid+1, target);
				if (idx != nums.end() && *idx == target)
					return idx - nums.begin();
				else {
					return -1;
				}
			}
			else {
				first = mid;
			}
		}
	}
	if (nums[0] <= target) {
		auto first = nums.begin();
		auto sec = nums.end() - 1;
		while (first != sec) {
			if (first - sec <= 1) {
				for (auto it = first;it <= sec;it++) {
					if (*it == target) {
						return it - nums.begin();
					}
				}
				return -1;
			}
			auto half = (sec - first + 1) / 2;
			auto mid = first + half;
			if (*mid == target)
				return mid - nums.begin();
			if (*mid > target) {
				auto idx = lower_bound(first, mid + 1, target);
				if (idx != nums.end() && *idx == target)
					return idx - nums.begin();
				else {
					return -1;
				}
			}
			else {
				first = mid;
			}
		}
	}
	else {
		if (nums[nums.size() - 1] < target)
			return -1;
		auto first = nums.begin();
		auto sec = nums.end() - 1;
		while (first != sec) {
			if (first - sec <= 1) {
				for (auto it = first;it <= sec;it++) {
					if (*it == target) {
						return it - nums.begin();
					}
				}
				return -1;
			}
			auto half = (sec - first + 1) / 2;
			auto mid = first + half;
			if (*mid == target)
				return mid - nums.begin();
			if (*mid < target) {
				auto idx = lower_bound(mid ,sec+1, target);
				if (idx != nums.end())
					return idx - nums.begin();
				else {
					return -1;
				}
			}
			else {
				sec = mid;
			}
		}
	}
}
#pragma endregion

#pragma region 34. Find First and Last Position of Element in Sorted Array
vector<int> searchRange(vector<int>& nums, int target) {
	auto lower = lower_bound(nums.begin(), nums.end(), target);
	auto upper = upper_bound(nums.begin(), nums.end(), target);
	vector<int> result;
	if (lower != nums.end()) {
		if (*lower == target) {
			result.push_back(lower - nums.begin());
			result.push_back(upper - nums.begin() - 1);
			return result;
		}
	}
	result.push_back(-1);
	result.push_back(-1);
	return result;
}
#pragma endregion

#pragma region 35. Search Insert Position
int searchInsert(vector<int>& nums, int target) {
	auto lower = lower_bound(nums.begin(), nums.end(), target);
	return lower - nums.begin();
}
#pragma endregion


#pragma region 41. First Missing Positive
int firstMissingPositive(vector<int>& nums) {
	int len = nums.size();
	for (int i = 0;i < len;i++) {
		if (nums[i] < 0)
			nums[i] = 0;
	}
	for (int i = 0;i < len;i++) {
		if (abs(nums[i]) <= len && abs(nums[i]) > 0) {
			if(nums[abs(nums[i]) - 1]>0)
					nums[abs(nums[i])-1]= -nums[abs(nums[i])-1];
			else if (nums[abs(nums[i]) - 1] == 0)
				nums[abs(nums[i]) - 1] = -len-10;
		}
	}
	int i = 0;
	for(;i < len;i++)
	{
		if (nums[i] >= 0)
			break;
	}
	return i + 1;
}
#pragma endregion

#pragma region 42. Trapping Rain Water
int findmax(vector<int>& height, int p1,int p2) {
	if (p2 < p1)
		return -1;
	if (p2 == p1)
		return p1;
	int max = 0;int maxIdx = p1;
	for (int i = p1;i <= p2;i++) {
		if (height[i]>=max) {
			max = height[i];
			maxIdx = i;
		}
	}
	return maxIdx;
}
int leftsubtrap(vector<int>& height,int p1,int p2) {
	if (p1 == p2)
		return 0;
	int submaxIdx = p2;
	while (submaxIdx > 0&&height[p2]== height[submaxIdx] ) {//search from the first different value 
		submaxIdx--;
	}
	submaxIdx++;// note height[submaxIdx] == height[p2] here
	if (submaxIdx == 0)// means all left values are equal
		return 0;
	int mid = findmax(height, p1, submaxIdx-1);
	if (mid == 0 && submaxIdx == 1)
		return 0;
	int mi = min(height[mid], height[p2]);
	int sum = 0;
	for (int i = mid+1;i != submaxIdx;i++) {
		sum += mi-height[i];
	}
	sum = leftsubtrap(height, p1, mid) + sum;
	return sum;
}
int rightsubtrap(vector<int>& height, int p1, int p2) {
	if (p1 == p2)
		return 0;
	int submaxIdx = p1;
	while (submaxIdx < height.size() && height[p1] == height[submaxIdx]) {//search from the first different value 
		submaxIdx++;
	}
	submaxIdx--;// note height[submaxIdx] == height[p1] here
	if (submaxIdx == height.size())// means all right values are equal
		return 0;
	int mid = findmax(height, submaxIdx+1, p2);
	if (mid == height.size()-1 && submaxIdx == height.size() - 2)
		return 0;

	int mi = min(height[mid], height[p1]);
	int sum = 0;
	for (int i = submaxIdx +1;i != mid;i++) {
		sum += mi - height[i];
	}
	sum = rightsubtrap(height, mid, p2) + sum;
	return sum;
}

int trap(vector<int>& height) {
	int len = height.size();
	if (len <= 2)
		return 0;
	int p1 = 0;int p2 = len - 1;
	int mid = findmax(height, p1, p2);
	int sum = leftsubtrap(height, p1, mid) + rightsubtrap(height,mid,p2);
	return sum;
}
#pragma endregion

#pragma region 46. ȫ����
//swapIntVector is already defined
int subPermute(vector<int>& nums,int first,vector<vector<int>>&res) {
	if (first == nums.size() - 1) 		{
		res.push_back(nums);
	}
	int secd = first;
	while (secd!=nums.size()) {
		swapIntVector(nums.begin()+first,nums.begin()+secd);
		subPermute(nums, first + 1, res);
		swapIntVector(nums.begin() + first, nums.begin() + secd);
		secd++;
	}
	return 0;
}
vector<vector<int>> permute(vector<int>& nums) {
	if (nums.size() == 1)
		return vector<vector<int>>{nums};
	vector<vector<int>> res;
	subPermute(nums,0,res);
	return res;
}
#pragma endregion


#pragma region 48. Rotate Image
void rotate(vector<vector<int>>& matrix) {

}
#pragma endregion

#pragma region 53. Maximum Subarray
#pragma region my stupid method

//int subMaxSubArray(vector<int>& nums, int p1) {
//	if (p1 == nums.size())
//		return 0;
//	if (p1 == nums.size() - 1)
//		return nums[p1];
//	if (nums[p1] + nums[p1 + 1] > 0) {
//		int consVal = nums[p1] + nums[p1 + 1];
//		p1 += 2;
//		while (p1<nums.size()-1) {
//			if (nums[p1] + nums[p1 + 1] > 0) {
//				consVal += nums[p1] + nums[p1 + 1];
//				p1 += 2;
//			}
//			else {
//				consVal += nums[p1];break;
//			}
//		}
//		return max(consVal, subMaxSubArray(nums, p1));
//	}
//	else {
//		return max(nums[p1], subMaxSubArray(nums, p1 + 2));
//	}
//}
//int maxSubArray(vector<int>& nums) {
//	if (nums.size() == 1)
//		return nums[0];
//	int p1 = 0;	int posval = 0;	int ngval = 0;
//	int max = INT_MIN;
//	while (p1 < nums.size()&&nums[p1] <= 0) {
//		if (nums[p1] > max)
//			max = nums[p1];
//		p1++;
//	}
//	if (p1 == nums.size())// all less or equal 0
//		return max;
//	vector<int> compressed;
//	while (p1<nums.size()) {
//		while (p1 <= nums.size() - 1 && nums[p1] >= 0) {
//			posval += nums[p1];
//			p1++;
//		}
//		if (posval > 0) {
//			compressed.push_back(posval);posval = 0;
//		}
//		while (p1 <= nums.size() - 1 && nums[p1] <= 0) {
//			ngval += nums[p1];
//			p1++;
//		}
//		if (ngval < 0) 			{
//			compressed.push_back(ngval);ngval = 0;
//		}
//	}
//	if (compressed[compressed.size() - 1] <= 0)
//		compressed.pop_back();
//	return subMaxSubArray(compressed,0);
//}
#pragma endregion

int maxSubArray(vector<int>& nums) {
	if (nums.size() == 1)
		return nums[0];
	int sum = 0;int res = nums[0];
	for (int i = 0;i < nums.size();i++) {
		sum += nums[i];
		res = max(sum,res);
		if (sum < 0)
			sum = 0;
	}
	return res;
}
#pragma endregion

#pragma region 54. Spiral Matrix
vector<int> readHorizontal(vector<vector<int>>& matrix, pair<int, int> start, pair<int, int> end) {
	assert(start.first==end.first);
	vector<int> result;
	if (start.second > end.second)
		return result;
	for (int i = start.second;i <= end.second;i++) {
		result.push_back(matrix[start.first][i]);
	}
	return result;
}
vector<int> readHorizontalInverse(vector<vector<int>>& matrix, pair<int, int> start, pair<int, int> end) {
	assert(start.first == end.first);
	vector<int> result;
	if (start.second > end.second)
		return result;
	for (int i = end.second;i >= start.second;i--) {
		result.push_back(matrix[start.first][i]);
	}
	return result;
}
vector<int> readVertical(vector<vector<int>>& matrix, pair<int, int> start, pair<int, int> end) {
	assert(start.second == end.second);
	vector<int> result;
	if (start.first > end.first)
		return result;
	for (int i = start.first;i <= end.first;i++) {
		result.push_back(matrix[i][start.second]);
	}
	return result;
}
vector<int> readVerticalInverse(vector<vector<int>>& matrix, pair<int, int> start, pair<int, int> end) {
	assert(start.second == end.second);
	vector<int> result;
	if (start.first > end.first)
		return result;
	for (int i = end.first;i >= start.first;i--) {
		result.push_back(matrix[i][start.second]);
	}
	return result;
}

vector<int> readCircle(vector<vector<int>>& matrix,pair<int,int> start,pair<int,int> end) {
	vector<int> result;
	if (start.first == end.first) {
		result = readHorizontal(matrix,start,end);
	} else if(start.second == end.second) {
		result = readVertical(matrix, start, end);
	} else {
		result = readHorizontal(matrix, start, pair<int, int>(start.first, end.second));
		auto tmp1 = readVertical(matrix, pair<int,int>(start.first+1, end.second), end);
		auto tmp2 = readHorizontalInverse(matrix, pair<int, int>(end.first, start.second), pair<int, int>(end.first, end.second-1));
		auto tmp3 = readVerticalInverse(matrix, pair<int, int>(start.first+1, start.second), pair<int, int>(end.first-1, start.second));

		result.insert(result.end(), tmp1.begin(), tmp1.end());
		result.insert(result.end(), tmp2.begin(), tmp2.end());
		result.insert(result.end(), tmp3.begin(), tmp3.end());
	}
	return result;
}
vector<int> spiralOrder(vector<vector<int>>& matrix) {
	vector<int> result;
	if (matrix.size() == 0 || matrix[0].size() == 0)
		return result;
	pair<int, int> start(0, 0);
	pair<int, int> end(matrix.size()-1, matrix[0].size()-1);
	while (start.first<=end.first&&start.second<=end.second) {
		auto tmp = readCircle(matrix, start, end);
		start.first++;start.second++;end.first--;end.second--;
		result.insert(result.end(), tmp.begin(), tmp.end());
	}
	return result;
}

#pragma endregion



#pragma region 55. Jump Game
bool canJump(vector<int>& nums) {
	if (nums.size() == 1 && nums[0] == 0)
		return true;
	int idx = 0;
	int p1 = idx;
	while (idx !=nums.size()) {
		if (nums[idx] != 0)
		{
			idx++;
		}
		else {
			int i = idx - 1;
			for (;i >= 0;i--) {
				int minsteps = idx - i + 1;
				if (idx == nums.size() - 1) {
					minsteps = idx - i ;
				}
				if (nums[i] >= minsteps) {
					idx++;
					p1 = idx;
					break;
				}
			}
			if (i == -1)
				return false;
		}
	}
	return true;
}
#pragma endregion

#pragma region 58. Length of Last Word
int lengthOfLastWord(string s) {
	if (s.size() == 1 && s[0] == ' ')
		return 0;
	if (s.size() == 1 && s[0] != ' ')
		return 1;
	auto p1 = s.rbegin();
	while (*p1 == ' ') {
		p1++;
	}
	auto p2 = p1;
	for (;p1 != s.rend();p1++) {
		if (*p1 == ' ')
			break;
	}
	return p1 - p2;
}
#pragma endregion

#pragma region 66. Plus One
vector<int> plusOne(vector<int>& digits) {
	bool flow = true;
	for (int i = digits.size() - 1;i > -1;i--) {
		if (flow == true) {
			digits[i] += 1;
			if (digits[i] == 10) {
				flow = true;
				digits[i] = 0;
			}
			else {
				flow = false;
			}
		}
		else {
			break;
		}
	}
	if (flow)
		digits.insert(digits.begin(), 1);
	return digits;
}
#pragma endregion


#pragma region 347. Top K Frequent Elements 
#pragma region With RbTree
//struct myLess {// can not use class declaration
//	bool operator()(pair<int, int>l1, pair<int, int>l2)const {
//		return l1.second < l2.second;
//	}
//};
//vector<int> topKFrequent(vector<int>& nums, int k) {
//	unordered_map<int, int> m;
//	for (int i = 0;i < nums.size();i++) {
//		m[nums[i]] = 0;
//	}
//	for (int i = 0;i < nums.size();i++) {
//		m[nums[i]] ++;
//	}
//	multiset<pair<int, int>, myLess> ps;
//	int min;
//	for (int i = 0;i < nums.size();i++) {
//		pair<int, int> tmp;
//		if (m[nums[i]] != -1) {
//			tmp.first = nums[i];tmp.second = m[nums[i]];
//			m[nums[i]] = -1;// mark that we have put it in queue
//			if (ps.size() >= k) {
//				min = (*ps.begin()).second;
//				if (tmp.second > min) {
//					ps.erase(ps.begin());
//					ps.insert(tmp);
//				}
//				else {/*do nothing*/ }
//			}
//			else {
//				ps.insert(tmp);
//			}
//		}
//	}
//	vector<int> result;
//	for (auto it = ps.begin();it != ps.end();it++) {
//		result.push_back((*it).first);
//	}
//	return result;
//}
#pragma endregion

#pragma region With Priority_queue
struct myGreater {// can not use class declaration
	bool operator()(pair<int, int>l1, pair<int, int>l2)const {
		return l1.second > l2.second;
	}
};
vector<int> topKFrequent(vector<int>& nums, int k) {
	unordered_map<int, int> m;
	for (int i = 0;i < nums.size();i++) {
		m[nums[i]] = 0;
	}
	for (int i = 0;i < nums.size();i++) {
		m[nums[i]] ++;
	}
	priority_queue<pair<int, int>, vector<pair<int, int>>,myGreater> ps;
	int min;
	for (int i = 0;i < nums.size();i++) {
		pair<int, int> tmp;
		if (m[nums[i]] != -1) {
			tmp.first = nums[i];tmp.second = m[nums[i]];
			m[nums[i]] = -1;// mark that we have put it in queue
			if (ps.size() >= k) {
				min = ps.top().second;
				if (tmp.second > min) {
					ps.pop();
					ps.push(tmp);
				}
				else {/*do nothing*/ }
			}
			else {
				ps.push(tmp);
			}
		}
	}
	vector<int> result;
	for (int i = 0;i <k;i++) {
		pair<int, int> tmp;tmp = ps.top();ps.pop();
		result.push_back(tmp.first);
	}
	return result;
}
#pragma endregion
#pragma endregion



#pragma region 904. Fruit Into Baskets
bool canbePick(int fruit_tyte,int& bucket_type1,int& bucket_type2) {
	if (bucket_type1 != bucket_type2) {
		return (fruit_tyte == bucket_type1 || fruit_tyte == bucket_type2);
	}
	else {
		bucket_type1 = fruit_tyte;
	}
	return true;
}
int totalFruit(vector<int>& fruits) {
	if (fruits.size() <= 2)
		return fruits.size();
	int result=2;
	int bucket_type1 = fruits[0];
	int bucket_type2 = fruits[1];
	int local_largest = result;
	int nearest_first_idx = 1;
	for (int i = 2;i < fruits.size();i++) {
		if (canbePick(fruits[i], bucket_type1, bucket_type2)) {
			local_largest++;
			if (fruits[i] != fruits[i - 1])
				nearest_first_idx = i;
		} else {
			if (local_largest > result) {
				result = local_largest;
			}
			bucket_type1 = fruits[nearest_first_idx];
			bucket_type2 = fruits[i];
			local_largest = i-nearest_first_idx+1;
			nearest_first_idx = i;
		}
	}
	if (local_largest > result)
		result = local_largest;
	return result;
}

#pragma endregion










