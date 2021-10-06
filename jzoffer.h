#pragma once
#include<algorithm>
#include<vector>
#include<unordered_map>
#include<list>
#include<stack>
using namespace std;

struct TreeNode {
	int val;
	TreeNode* left;
	TreeNode* right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}
};
class Node {
public:
    int val;
    Node* next;
    Node* random;
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};
class CQueue {
    stack<int> sk1, sk2;
public:
    CQueue() {
        while (!sk1.empty()) {
            sk1.pop();
        }
        while (!sk2.empty()) {
            sk2.pop();
        }
    }

    void appendTail(int value) {
        sk1.push(value);
    }

    int deleteHead() {
        if (sk2.empty()) {
            while (!sk1.empty()) {
                sk2.push(sk1.top());
                sk1.pop();
            }
        }
        if (sk2.empty()) {
            return -1;
        }
        else {
            int deleteItem = sk2.top();
            sk2.pop();
            return deleteItem;
        }
    }
};
class MinStack {
    stack<int> x_stack;
    stack<int> min_stack;
public:
    MinStack() {
        min_stack.push(INT_MAX);
    }

    void push(int x) {
        x_stack.push(x);
        min_stack.push(std::min(min_stack.top(), x));
    }

    void pop() {
        x_stack.pop();
        min_stack.pop();
    }

    int top() {
        return x_stack.top();
    }

    int min() {
        return min_stack.top();
    }
};

class MaxQueue {
    int mqueue[100001] = { 0 };
    int help[100001] = { 0 };
    int head = 0;int tail = -1;
    int maxhead = 0;int maxtail = -1;
public:
    MaxQueue() {

    }

    int max_value() {
        return help[maxhead];
    }

    void push_back(int value) {
        while (maxtail > maxhead - 1 && value > help[maxtail]) {
            maxtail--;
        }
        maxtail++;
        maxtail = maxtail % 100000;
        help[maxtail] = value;

        tail++;
        tail = tail % 100000;
        mqueue[tail] = value;
    }

    int pop_front() {
        if (tail < head)
            return -1;
        if (mqueue[head] == help[maxhead]) {
            maxhead++; maxhead = maxhead % 100000;
        }
        int tmp = mqueue[head];
        head++;
        head = head % 100000;
        return tmp;
    }
};
namespace jzoffer {
    string replaceSpace(string s);
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target);
    int movingCount(int m, int n, int k);
    vector<int> spiralOrder(vector<vector<int>>& matrix);
    bool validateStackSequences(vector<int>& pushed, vector<int>& popped);
    vector<string> permutation(string s);
    int maxValue(vector<vector<int>>& grid);
    int reversePairs(vector<int>& nums);
    int kthLargest(TreeNode* root, int k);
    string reverseWords(string s);
    int add(int a, int b);
    int lastRemaining(int n, int m);
    vector<vector<int>> findContinuousSequence(int target);
    vector<int> constructArr(vector<int>& a);
    vector<int> maxSlidingWindow(vector<int>& nums, int k);
    int sumNums(int n);
}
