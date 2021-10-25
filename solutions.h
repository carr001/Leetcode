#pragma once
#include<algorithm>
#include<vector>
#include<unordered_map>
#include<list>
using namespace std;

vector<int> twoSum(vector<int>& nums, int target);

struct ListNode {
	int val;
	ListNode* next;
	ListNode() : val(0), next(nullptr) {}
	ListNode(int x) : val(x), next(nullptr) {}
	ListNode(int x, ListNode* next) : val(x), next(next) {}
};
ListNode* addTwoNumbers(ListNode* l1, ListNode* l2);
int lengthOfLongestSubstring(string s);
string longestPalindrome(string s);
bool is_palindromic(string s, int start, int end);
string convert(string s, int numRows);
int reverse(int x);
int myAtoi(string s);
bool isPalindrome(int x);
bool isMatch(string s, string p);
int maxArea(vector<int>& height);
string longestCommonPrefix(vector<string>& strs);
vector<vector<int>> threeSum(vector<int>& nums);
vector<vector<int>> twoSum2(vector<int>nums, int target);
vector<string> letterCombinations(string digits);
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);
vector<int> topKFrequent(vector<int>& nums, int k);
ListNode* mergeKLists(vector<ListNode*>& lists);
ListNode* swapPairs(ListNode* head);
vector<int> findSubstring(string s, vector<string>& words);
void nextPermutation(vector<int>& nums);
int search(vector<int>& nums, int target);

int firstMissingPositive(vector<int>& nums);
int trap(vector<int>& height);
int maxSubArray(vector<int>& nums);
bool canJump(vector<int>& nums);
vector<vector<int>> permute(vector<int>& nums);
vector<int> plusOne(vector<int>& digits);
vector<int> spiralOrder(vector<vector<int>>& matrix);




int totalFruit(vector<int>& fruits);