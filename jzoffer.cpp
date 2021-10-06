

#include "jzoffer.h"
#include"solutions.h"
#include<algorithm>
#include<math.h>
#include<vector>
#include<unordered_set>
#include<list>
#include<array>
#include<assert.h>
#include<regex>
#include<iostream>
#include<sstream>
#include< queue >
#include<map>
#include<set>
#include<unordered_set>
#include<stack>
using namespace std;
namespace jzoffer{

#pragma region  03. 数组中重复的数字
    // key：判断unordered_set中是否包含某元素的方法
    int findRepeatNumber(vector<int>& nums) {
        unordered_set<int> usets;
        for (int i = 0;i < nums.size();i++) {
            unordered_set<int>::iterator res = usets.find(nums[i]);
            if (res != usets.end())
                return nums[i];
            else {
                usets.insert(nums[i]);
            }
        }
        return 0;
    }
#pragma endregion
#pragma region  04. 二维数组中的查找
#pragma region my stupid solution
    //pair<int,int> searchDiag(vector<vector<int>>& matrix, int target, int idx1_r, int idx1_c, int idx2_r, int idx2_c) {
    //    if (idx1_r == idx2_r|| idx1_c == idx2_c) {
    //        return pair<int, int>(idx1_r, idx1_c);
    //    }
    //    int first_r = idx1_r;
    //    int first_c = idx1_c;
    //    int half = min(idx2_r - idx1_r, idx2_c - idx1_c);
    //    int sec_r = first_r+half;
    //    int sec_c = first_c + half;
    //    if (half != 1)
    //        half = half >> 1;
    //    int mid_r = first_r + half;
    //    int mid_c = first_c + half;
    //    while ((first_r != sec_r)) {
    //        if (matrix[mid_r][mid_c] == target || ((mid_r + 1 < matrix.size()&& mid_c + 1 < matrix[0].size()) && matrix[mid_r + 1][mid_c+1] > target && matrix[mid_r][mid_c] < target))
    //            return pair<int, int>(mid_r, mid_c);
    //        if (matrix[mid_r][mid_c] < target) {
    //            first_r = mid_r;
    //            first_c = mid_c;
    //            if(half!=1)
    //                half = min(sec_r - first_r, sec_c - first_c) >> 1;
    //            else {
    //                mid_r = first_r;
    //                mid_c = first_c;
    //            }
    //            mid_r = first_r + half;
    //            mid_c = first_c + half;
    //        }
    //        else {
    //            half = half >> 1;
    //            sec_r = mid_r;
    //            sec_c = mid_c;
    //            mid_r = first_r + half;
    //            mid_c = first_c + half;
    //        }
    //    }
    //    return pair<int, int>(mid_r, mid_c);
    //}

    //int searchIthCol(vector<vector<int>>& matrix, int ithCol, int target) {
    //    int first = 0;
    //    int sec = matrix.size()-1;
    //    int half = sec - first;
    //    if (sec - first != 1)
    //        half = (sec - first) >> 1;
    //    int mid = first + half;
    //    while (sec != first) {
    //        if (matrix[mid][ithCol] == target||((mid+1< matrix.size())&&matrix[mid+1][ithCol]>target&& matrix[mid][ithCol] < target))
    //            return mid;
    //        if (matrix[mid][ithCol] < target) {
    //            first = mid;
    //            if (sec - first != 1)
    //                half = (sec - first) >> 1;
    //            else {
    //                sec = first;
    //            }
    //            mid = first + half;
    //        }
    //        else {
    //            half = half >> 1;
    //            sec = mid;
    //            mid = first + half;
    //        }
    //    }
    //    return mid;
    //}
    //int searchIthRow(vector<vector<int>>& matrix,int ithRow,int target) {
    //    auto lower = std::lower_bound(matrix[ithRow].begin(), matrix[ithRow].end(), target);
    //    if (lower != matrix[ithRow].end()) {
    //        if (matrix[ithRow][lower - matrix[ithRow].begin()] == target) {
    //            return lower - matrix[ithRow].begin();
    //        }
    //        else if (lower == matrix[ithRow].begin())
    //            return 0;
    //        else {
    //            return lower - matrix[ithRow].begin() - 1;
    //        }
    //    }
    //    else {
    //        return lower - matrix[ithRow].begin()-1;
    //    }
    //}

    //bool findNumberIn2DArraySubproblem(vector<vector<int>>& matrix,int target,int idx1_r,int idx1_c,int idx2_r,int idx2_c) {
    //    if(idx1_r==idx2_r)
    //        return (matrix[searchIthCol(matrix, idx1_c, target)][idx1_c] == target);
    //    if (idx1_c==idx2_c) {
    //        return (matrix[idx1_r][searchIthRow(matrix, idx1_r, target)] == target);
    //    }
    //    pair<int, int> idx = searchDiag(matrix, target, idx1_r, idx1_c, idx2_r, idx2_c);
    //    if (matrix[idx.first][idx.second] == target)
    //        return true;
    //    if ((idx.first == idx1_r)) {// the first elem is diag
    //        bool res1 = (matrix[idx1_r][searchIthRow(matrix, idx1_r, target)] == target);
    //        if (res1 == true)
    //            return true;
    //        else {
    //            return (matrix[searchIthCol(matrix, idx1_c, target)][idx1_c] == target);
    //        }
    //    }
    //    bool res1 = findNumberIn2DArraySubproblem(matrix, target, idx.first, idx1_c, idx2_r, idx.second);
    //    bool res2 = findNumberIn2DArraySubproblem(matrix, target, idx1_r, idx.second, idx.first, idx2_c);
    //    return res1 || res2;
    //}
    //bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
    //    if (matrix.size() == 0|| matrix[0].size() == 0)
    //        return false;
    //    int idx1_r = 0;int idx1_c = 0;
    //    int idx2_r = matrix.size()-1;int idx2_c = matrix[0].size()-1;
    //    return findNumberIn2DArraySubproblem(matrix, target, idx1_r, idx1_c,idx2_r,idx2_c);
    //}
#pragma endregion
    
// copy from https://leetcode-cn.com/problems/er-wei-shu-zu-zhong-de-cha-zhao-lcof/solution/mian-shi-ti-04-er-wei-shu-zu-zhong-de-cha-zhao-zuo/
// bravo
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        int i = matrix.size() - 1, j = 0;
        while (i >= 0 && j < matrix[0].size()) {
            if (matrix[i][j] > target) i--;
            else if (matrix[i][j] < target) j++;
            else return true;
        }
        return false;
    }
#pragma endregion

#pragma region 05. 替换空格
    string replaceSpace(string s) {
        int len = s.length();
        for (int i = 0;i != len;i++) {
            if (s[i] == ' ') {
                s[i] = '%';
                s.insert(s.begin() + i + 1, '0');
                s.insert(s.begin() + i + 1, '2');
                len = s.length();
            }
        }
        return s;
    }
#pragma endregion

#pragma region 06. 从尾到头打印链表
vector<int> reversePrint(ListNode* head) {
    /*if (head == NULL) {
        return vector<int>();
    }*/
    stack<ListNode*> sk;
    ListNode* p = head;
    while (p != NULL) {
        sk.push(p);
        p = p->next;
    }
    vector<int> res;
    while (!sk.empty()) {
        ListNode* tp = sk.top();
        res.push_back(tp->val);
        sk.pop();
    }
    return res;
}
#pragma endregion

#pragma region 07. 重建二叉树

    TreeNode* subBuildTree(vector<int>& preorder, vector<int>& inorder, int pp1, int pp2, int ip1, int ip2) {
        if (pp1 > pp2 || ip1 > ip2)
            return NULL;
        assert((ip2 - ip1) == (pp2 - pp1));
        TreeNode* root = (TreeNode*)malloc(sizeof(TreeNode));
        root->val = preorder[pp1];
        root->left = NULL;
        root->right = NULL;
        if (pp1 == pp2)
            return root;
        int left = ip1;
        while (inorder[left] != root->val) {
            left++;
        }
        root->left = subBuildTree(preorder, inorder, pp1 + 1, pp1 + left - ip1, ip1, left - 1);
        root->right = subBuildTree(preorder, inorder, pp1 + left - ip1 + 1, pp2, left + 1, ip2);
        return root;
    }
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        TreeNode* res = subBuildTree(preorder, inorder, 0, preorder.size() - 1, 0, inorder.size() - 1);
        return res;
    }
#pragma endregion

#pragma region 09. 用两个栈实现队列

#pragma endregion

#pragma region 10- I. 斐波那契数列
int fibArray[101] = {0};
int fib(int n) {
    if (n == 0) {
        fibArray[n] = 0;
        return 0;
    }
    if (fibArray[n] != 0)
        return fibArray[n];
    if (n == 1) {
        fibArray[n] = 1;
        return 1;
    }
    int res = (fib(n - 1) + fib(n - 2))% 1000000007;
    fibArray[n] = res;
    return res ;
}
//下面只适合f(0)=0,f(1)=1的情况
//int fib(int n) {
//    double num = pow((1 + sqrt(5)) / 2, n) - pow((1 - sqrt(5)) / 2, n);
//    int res = num / sqrt(5);
//    return res % 1000000007;
//}
#pragma endregion
#pragma region 10- II. 青蛙跳台阶问题
//同斐波那契，关键的问题是，将求解numWays(n)转化为求解numWays(n-1)与numWays(n-2)
int numWays(int n) {
    if (n <= 1) return 1;
    int dp[100 + 1];
    dp[0] = 1;
    dp[1] = 1;
    //dp[2] = 2;

    for (int i = 2; i < n + 1; i++) {
        dp[i] = (dp[i - 1] + dp[i - 2]) % 1000000007;
    }
    return dp[n];
}
#pragma endregion

#pragma region 12. 矩阵中的路径

int rows, cols;
bool dfs(vector<vector<char>>& board, string word, int i, int j, int k) {
    if (i >= rows || i < 0 || j >= cols || j < 0 || board[i][j] != word[k]) return false;
    if (k == word.size() - 1) return true;
    board[i][j] = '\0';
    bool res = dfs(board, word, i + 1, j, k + 1) || dfs(board, word, i - 1, j, k + 1) ||
        dfs(board, word, i, j + 1, k + 1) || dfs(board, word, i, j - 1, k + 1);
    board[i][j] = word[k];
    return res;
}

bool exist(vector<vector<char>>& board, string word) {
    rows = board.size();
    cols = board[0].size();
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (dfs(board, word, i, j, 0)) return true;
        }
    }
    return false;
}
#pragma endregion

#pragma region 13. 机器人的运动范围
int movingCount(int m, int n, int k) {
    vector<vector<bool>> visited(m, vector<bool>(n, 0));
    queue<vector<int>> queue;queue.push({ 0,0,0,0 });
    int count = 0;
    while (queue.size()>0) {
        vector<int> front = queue.front();
        queue.pop();
        int i = front[0];int j = front[1]; int s1 = front[2];int s2 = front[3];
        if (i >= m || j >= n ||s1+s2>k ||visited[i][j] == true) continue;
        count++;
        visited[i][j] = true;
        queue.push({i+1,j,(s1+1)%10==0?s1-8:s1+1,s2});
        queue.push({i,j+1,s1,(s2 + 1) % 10 == 0 ? s2 - 8 : s2 + 1 });
    }
    return count;
}
#pragma endregion


#pragma region 15. 二进制中1的个数
int hammingWeight(uint32_t n) {
    int c = 0;
    while (n) {
        if (n % 2 == 1) c++;
        n = n >> 1;
    }
    return n;
}
#pragma endregion

#pragma region 16. 数值的整数次方
double myPow(double x, int n) {
        bool minus = false;
        unsigned int k = n;
        if (n < 0) {
            minus = true;
            k = abs(n);
        }
        double res = 1;
        while (k) {
            if (k % 2 == 1) {
                res = res * x; x = x * x;
                k = k >> 1;
            }
            else {
                x = x * x;
                k = k >> 1;
            }
        }
        if (minus) return 1 / res;
        return res;
}
#pragma endregion

#pragma region 17. 打印从1到最大的n位数
vector<int> printNumbers(int n) {
    int i = 1;int b=10;
    vector<int> res;
    while (n) {
        if (i % b == 0) {
            b = b * 10;n--;
        }
        res.push_back(i);
    }
    return res;
}
#pragma endregion

#pragma region 18. 删除链表的节点
ListNode* deleteNode(ListNode* head, int val) {
    if (head == NULL)
        return NULL;
    if (val == head->val) {
        if (head->next == NULL)
            return NULL;
        else {
            head = head->next;
            return head;
        }
    }
    ListNode* p1 = head;
    ListNode* pre = head;
    while (p1->val!=val&&p1!=NULL) {
        pre = p1;p1 = p1->next;
    }
    if (p1 == NULL)
        return head;
    else {
        pre->next = p1->next;
        return head;
    }
}

#pragma endregion

#pragma region  26. 树的子结构
bool isTreeContain(TreeNode* A, TreeNode* B) {
    if (B == NULL && A == NULL || B == NULL && A != NULL) {
        return true;
    }
    else if (B != NULL && A == NULL)
        return false;
    if (A->val == B->val)
        return isTreeContain(A->left, B->left) && isTreeContain(A->right, B->right);
    else {
        return false;
    }
    return false;
}

bool isSubStructure(TreeNode* A, TreeNode* B) {
    if (B == NULL && A == NULL) {
        return true;
    }
    else if (B == NULL && A != NULL) {
        return false;
    }
    if (A == NULL) {
        return false;
    }
    if (A->val == B->val) {
        if (isTreeContain(A, B)) {
            return true;
        }
        else {
            return isSubStructure(A->left, B) || isSubStructure(A->right, B);
        }
    }
    else {
        return isSubStructure(A->left, B) || isSubStructure(A->right, B);
    }
    return false;

}
#pragma endregion


#pragma region 27. 二叉树的镜像
TreeNode* mirrorTree(TreeNode* root) {
    if (root == NULL)
        return NULL;
    TreeNode* p1 = root->left;
    root->left = mirrorTree(root->right);

    root->right = mirrorTree(p1);
    return root;
}
#pragma endregion

#pragma region 28. 对称的二叉树
bool isSubTreeSymmetric(TreeNode* A, TreeNode* B) {
    if (B == NULL && A == NULL) {
        return true;
    }
    else if (B == NULL && A != NULL) {
        return false;
    }
    else if (B != NULL && A == NULL)
        return false;
    if (A->val == B->val)
        return isSubTreeSymmetric(A->left, B->right) && isSubTreeSymmetric(A->right, B->left);
    else {
        return false;
    }
    return false;
}

bool isSymmetric(TreeNode* root) {
    if (root == NULL)
        return true;
    return isSubTreeSymmetric(root->left, root->right);
}
#pragma endregion

#pragma region 29. 顺时针打印矩阵
int count;
pair<int, int> mGoLeft(vector<vector<int>>& matrix, vector<int>& res, pair<int, int> curr) {
    // 保证curr 一定有元素
    int cols = matrix[0].size();
    int rows = matrix.size();
    while (curr.second < cols && matrix[curr.first][curr.second] != INT_MIN) {
        res.push_back(matrix[curr.first][curr.second]);
        matrix[curr.first][curr.second] = INT_MIN;
        curr.second++;
        count--;
    }
    curr.second--;
    return curr;
}
pair<int, int> mGoDown(vector<vector<int>>& matrix, vector<int>& res, pair<int, int> curr) {
    // 保证curr 一定有元素
    int cols = matrix[0].size();
    int rows = matrix.size();
    while (curr.first < rows && matrix[curr.first][curr.second] != INT_MIN) {
        res.push_back(matrix[curr.first][curr.second]);
        matrix[curr.first][curr.second] = INT_MIN;
        curr.first++;
        count--;
    }
    curr.first--;
    return curr;
}
pair<int, int> mGoRight(vector<vector<int>>& matrix, vector<int>& res, pair<int, int> curr) {
    // 保证curr 一定有元素
    int cols = matrix[0].size();
    int rows = matrix.size();
    while (curr.second > -1 && matrix[curr.first][curr.second] != INT_MIN) {
        res.push_back(matrix[curr.first][curr.second]);
        matrix[curr.first][curr.second] = INT_MIN;
        curr.second--;
        count--;
    }
    curr.second++;
    return curr;
}
pair<int, int> mGoUp(vector<vector<int>>& matrix, vector<int>& res, pair<int, int> curr) {
    // 保证curr 一定有元素
    int cols = matrix[0].size();
    int rows = matrix.size();
    while (curr.first > -1 && matrix[curr.first][curr.second] != INT_MIN) {
        res.push_back(matrix[curr.first][curr.second]);
        matrix[curr.first][curr.second] = INT_MIN;
        curr.first--;
        count--;
    }
    curr.first++;
    return curr;
}
vector<int> spiralOrder(vector<vector<int>>& matrix) {
    enum Direct { goLeft, goDown, goRight, goUp };
    struct CircleQueue
    {
        queue<Direct> myQueue;
        CircleQueue() { myQueue.push(Direct::goLeft);myQueue.push(Direct::goDown);myQueue.push(Direct::goRight);myQueue.push(Direct::goUp); };
        int nextDirection() {
            Direct tmp = myQueue.front();
            myQueue.pop();
            myQueue.push(tmp);
            return tmp;
        };
    }myCircleQueue;
    vector<int> res;
    int cols = matrix[0].size();
    int rows = matrix.size();
    if (cols == 0 || rows == 0)
        return res;
    count = cols * rows;
    pair<int, int> curr = { 0,0 };
    while (count > 0) {
        switch (myCircleQueue.nextDirection()) {
        case goLeft:
            curr = mGoLeft(matrix, res, curr);
            curr.first++;break;
        case goDown:
            curr = mGoDown(matrix, res, curr);
            curr.second--;break;
        case goRight:
            curr = mGoRight(matrix, res, curr);
            curr.first--;break;
        case goUp:
            curr = mGoUp(matrix, res, curr);
            curr.second++;break;
        default:
            break;
        }
    }
    return res;
}
#pragma endregion

#pragma region 31. 栈的压入弹出序列
bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
    if (pushed.size() == 0)
        return true;
    int i = 0;
    while (i!= pushed.size()-1) {
        while (pushed[i] == popped.front()) {
            if (i != pushed.size() - 1) {
                pushed.erase(pushed.begin() + i);
                popped.erase(popped.begin());
                i--;if (i == -1) i = 0;
            }
            else {
                break;
            }
        }
        if (i < pushed.size() - 1) {
                i++;
        }
    }
    if (i == pushed.size()-1) {
        for (int k = i;k > -1;k--) {
            if (pushed[k] != popped[i - k])
                return false;
        }
    }
    return true;
}
#pragma endregion

#pragma region 32 - I. 从上到下打印二叉树
vector<int> levelOrder(TreeNode* root) {
    vector<int> res;
    if (root == NULL)
        return res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        TreeNode* t = q.front();q.pop();
        res.push_back(t->val);
        if (t->left != NULL)
            q.push(t->left);
        if (t->right != NULL)
            q.push(t->right);
    }
    return res;
}
#pragma endregion
    
#pragma region 32 - II. 从上到下打印二叉树 II
vector<vector<int>> levelOrder2(TreeNode* root) {
    vector<vector<int>> res;
    if (root == NULL)
        return res;
    queue<TreeNode*> q;
    
    q.push(root);
    while (!q.empty()) {
        int size = q.size();
        vector<int> subRes;
        while (size) {
            size--;
            TreeNode* t = q.front();q.pop();
            subRes.push_back(t->val);
            if (t->left != NULL)
                q.push(t->left);
            if (t->right != NULL)
                q.push(t->right);
        }
        res.push_back(subRes);
    }
    return res;
} 
#pragma endregion

#pragma region 32 - III. 从上到下打印二叉树 III
vector<vector<int>> levelOrder3(TreeNode* root) {
    vector<vector<int>> res;
    if (root == NULL)
        return res;
    queue<TreeNode*> q;
    int count = 1;
    q.push(root);
    while (!q.empty()) {
        int size = q.size();
        vector<int> subRes;
        stack<TreeNode*>s;// 可以不用栈，直接reverse
        while (size) {
            size--;
            TreeNode* t = q.front();q.pop();
            if(count%2==1)
                subRes.push_back(t->val);
            else {
                s.push(t);
            }
            if (t->left != NULL)
                q.push(t->left);
            if (t->right != NULL)
                q.push(t->right);
        }
        if (count % 2 == 0) {
            while (!s.empty()) {
                subRes.push_back(s.top()->val);s.pop();
            }
        }
        res.push_back(subRes);
        count++;
    }
    return res;
}
#pragma endregion
#pragma region  33. 二叉搜索树的后序遍历序列
//bool subVerifyPostorderLeft(vector<int>& postorder, int pp1, int pp2,int target) {
//    assert(pp1 <= pp2);
//    if (pp1 == pp2)
//        return true;
//    int firstBigger = pp1;
//    while (postorder[firstBigger]<postorder[pp2]) {
//        if (postorder[firstBigger] > target)
//            return false;
//        firstBigger++;
//    }
//    return subVerifyPostorderLeft(postorder, pp1, firstBigger - 1);
//}
//bool subVerifyPostorderRight(vector<int>& postorder, int pp1, int pp2, int target) {
//
//}
//bool subVerifyPostorder(vector<int>& postorder,int pp1,int pp2) {
//    assert(pp1<=pp2);
//    if (pp1 == pp2)
//        return true;
//
//    int firstBigger = pp1;
//    while (postorder[firstBigger]<postorder[pp2]) {
//        firstBigger++;
//    }
//    if (firstBigger == pp2)
//        return subVerifyPostorder(postorder,pp1,firstBigger-1);
//    else if(firstBigger == pp1){
//        int i = pp1;int j = pp1 + 1;
//        while (j <= pp2) {
//            if (postorder[i] < postorder[j]||postorder[i]<postorder[pp2])
//                return false;
//            i++;j++;
//        }
//        return true;
//    }
//    else {
//        return subVerifyPostorder(postorder, pp1, firstBigger - 1) && subVerifyPostorder(postorder, firstBigger, pp2 - 1);
//    }
//}
//bool verifyPostorder(vector<int>& postorder) {
//    if (postorder.size() <= 1)
//        return true;
//    return subVerifyPostorderLeft(postorder, 0, postorder.size()-1,INTMAX_MAX);
//}
#pragma endregion

#pragma region 34. 二叉树中和为某一值的路径
int pathSubSum(TreeNode* root,int target, vector<vector<int>>& res,vector<int>& subRes) {
    if (root->left == NULL && root->right == NULL) {
        if (root->val == target) {
            subRes.push_back(root->val);
            res.push_back(subRes);
            subRes.pop_back();
            return 1;
        }
        return 0;
    }
    target -= root->val;
    subRes.push_back(root->val);
    if (root->left != NULL) {
        pathSubSum(root->left, target, res, subRes);
    }
    if (root->right != NULL) {
        pathSubSum(root->right, target, res, subRes);
    }
    subRes.pop_back();
    return 0;
}
vector<vector<int>> pathSum(TreeNode* root, int target) {
    if (root == NULL)
        return vector<vector<int>>();
    vector<vector<int>> res;
    vector<int> subRes;
    pathSubSum(root,target,res,subRes);
    return res;
}
#pragma endregion


#pragma region 35. 复杂链表的复制
Node* copyRandomList(Node* head) {
    if (head == NULL)
        return NULL;
    unordered_map<Node*, Node*> h;
    Node* p1 = head;
    Node* headCopy = new Node(head->val);
    Node* p1Copy = headCopy;
    h[head] = headCopy;
    while (p1) {
        if (p1->random != NULL) {
            if (h.find(p1->random) == h.end()) {
                Node* tmp = new Node(p1->random->val);
                h[p1->random] = tmp;
            }
            p1Copy->random = h[p1->random];
        }
        Node* p2 = p1->next;
        if (p2 != NULL) {

            if (h.find(p2) == h.end()) {
                Node* tmp = new Node(p2->val);
                h[p2] = tmp;
            }
            p1Copy->next = h[p2];
            p1Copy = p1Copy->next;
        }
        p1 = p2;
    }
    return headCopy;
}
#pragma endregion

#pragma region 36. 二叉搜索树与双向链表

#pragma endregion

#pragma region 38. 字符串的排列
// similar to allArrangement in other.cpp
int subPermutation(vector<string>& res,char memo, unordered_set<string>& h) {
    int len = res.size();
    for (int i = 0;i < len;i++) {
        string origin;
        int sublen = res[i].size();
        for (int j = 0;j <= sublen;j++) {
            if (j == 0) {
                origin = res[i];
                res[i].insert(res[i].begin(),memo);
                h.insert(res[i]);
            }
            else {
                origin.insert(origin.begin()+j,memo);
                if (h.find(origin) == h.end()) {
                    res.push_back(origin);
                    h.insert(origin);
                }
                origin.erase(origin.begin()+j);
            }
        }
    }
    return 0;
}
vector<string> permutation(string s) {
    if (s.length()==1)
        return vector<string>{s};
    unordered_set<string> h;
    vector<string> res;
    for (int i = 0;i < s.length();i++) {
        if (i == 0) {
            res.push_back({ s[i] });
            h.insert(s.substr(i, 1));
        }
        else {
            subPermutation(res,s[i],h);
        }
    }
    return res;
}
#pragma endregion

#pragma region 39. 数组中出现次数超过一半的数字
int majorityElement(vector<int>& nums) {
    if (nums.size() == 1)
        return nums[0];
    unordered_map<int,int> h;
    for (int i = 0;i < nums.size();i++) {
        if (h.find(nums[i]) == h.end())
            h[nums[i]] = 1;
        else {
            h[nums[i]]++;
        }
        if (h[nums[i]] >= ceil(float(nums.size()) / 2))
            return nums[i];
    }
    return -1;
}
#pragma endregion

#pragma region 40. 最小的k个数
//struct myGreater {// can not use class declaration
//    bool operator()(int l1, int l2)const {
//        return l1 > l2;
//    }
//};
vector<int> getLeastNumbers(vector<int>& arr, int k) {
    if (k == 0)
        return vector<int>();
	priority_queue<int> ps;
	int ma;
	for (int i = 0;i < arr.size();i++) {
		if (ps.size() >= k) {
            ma = ps.top();
			if (arr[i] < ma) {
				ps.pop();
				ps.push(arr[i]);
			}
			else {/*do nothing*/ }
		}
		else {
			ps.push(arr[i]);
		}
	}
	vector<int> result;
	for (int i = 0;i <k;i++) {
		int tmp;tmp = ps.top();ps.pop();
		result.push_back(tmp);
	}
	return result;
}
#pragma endregion

#pragma region 45. 把数组排成最小的数
bool mIsStringBigger(string s1,string s2) {
    int len = min(s1.length(), s2.length());
    for (int i = 0;i < len;i++) {
        if (s1[i] > s2[i])
            return true;
        if (s1[i] < s2[i])
            return false;
    }
    if (s1.length() > s2.length())
        return true;
    return false;
}
string minNumber(vector<int>& nums) {
    if (nums.size() == 0)
        return "";
    string res;
    while (nums.size()!=1) {
        string mi="999999999999999999999999";
        int minIdx = 0;
        for (int i = 0;i < nums.size();i++) {
            if (mIsStringBigger(mi,to_string(nums[i]))) {
                mi = to_string(nums[i]);
            }
        }
        res.append(mi);
        nums.erase(nums.begin()+minIdx);
    }
    res.append(to_string(nums[0]));
    return res;
}
#pragma endregion


#pragma region 47. 礼物的最大价值
int memo47[201][201] = { 0 };
int subMaxValue(vector<vector<int>>& grid, int row, int col) {
    if (row < 0 || col < 0)
        return 0;
    if (row == 0 && col == 0)
        return grid[row][col];
    if (memo47[row][col] > 0)
        return memo47[row][col];
    memo47[row][col] = grid[row][col]+max(subMaxValue(grid,row-1,col),subMaxValue(grid,row,col-1));
    return memo47[row][col];
}
int maxValue(vector<vector<int>>& grid) {
    int rows = grid.size();
    if (rows == 0)
        return 0;
    int cols = grid[0].size();
    if (cols == 0)
        return 0;
    return subMaxValue(grid,rows-1,cols-1);
}
#pragma endregion


#pragma region 50. 第一个只出现一次的字符
char firstUniqChar(string s) {
    if (s.size() == 0)
        return ' ';
    if (s.size() == 1)
        return s[0];
    unordered_map<char,int> h;
    for (char c:s) {
        if (h.find(c) == h.end())
            h[c] = 1;
        else {
            h[c] ++;
        }
    }for (char c : s) {
        if (h[c] == 1)
            return c;
    }
    return ' ';
}
#pragma endregion

#pragma region 51. 数组中的逆序对
int count51 = 0;
int mSwap(vector<int>& nums,int i,int j) {
    int tmp = nums[i];nums[i] = nums[j];nums[j] = tmp;return 0;
}
int mMergeAndCountInverse(vector<int>& nums, int p,int q) {
    if ((q - p) == 1) {
        if (nums[p] > nums[q]) {
            count51++;
            mSwap(nums,p,q);
        }
        return 0;
    }
    int r = (p + q) / 2;
    int i = p;
    int j = r + 1;
    while (i<r+1) {
        count51 += j - r - 1;
        while (j<=q) {
            if (nums[i] > nums[j]) {
                count51++;
                j++;
            }
            else {
                break;
            }
        }
        i++;
    }
    j = 0;i = 0;
    vector<int> L(nums.begin()+p,nums.begin()+r+1);
    vector<int> R(nums.begin()+r+1,nums.begin()+q+1);
    for (int k = p;k <= q;k++) {
        if (i < L.size() && j < R.size()) {
            if (L[i] <= R[j]) {
                nums[k] = L[i];
                i++;
            }
            else {
                nums[k] = R[j];
                j++;
            }
        }
        else 
        if (i < L.size()) {
            nums[k] = L[i];
            i++;
        }
        else
        if (j < R.size()) {
            nums[k] = R[j];
            j++;
        }
    }
    return 0;
}
int mergetSort(vector<int>& nums,int p,int q) {
    if (p == q)
        return nums[q];
    int r = (p + q) / 2;
    mergetSort(nums,p,r);
    mergetSort(nums, r + 1, q);
    mMergeAndCountInverse(nums, p,  q);
    return 0;
}
int reversePairs(vector<int>& nums) {
    count51 = 0;
    if (nums.size() <= 1)
        return count51;
    mergetSort(nums,0,nums.size()-1);
    return count51;
}
#pragma endregion

#pragma region 52. 两个链表的第一个公共节点
int getListLen(ListNode* l) {
    ListNode* p1 = l;
    int res=0;
    while (p1) {
        res++;
        p1 = p1->next;
    }
    return res;
}
ListNode* getIntersectionNode(ListNode* headA, ListNode* headB) {
    int len1 = getListLen(headA);
    int len2 = getListLen(headB);
    if (len1==0 || len2==0) {
        return NULL;
    }
    ListNode* p1 = headA;
    ListNode* p2 = headB;
    if (len1 > len2) {
        while (len1!=len2) {
            p1 = p1->next;
            len1--;
        }
    }else if (len1 < len2) {
        while (len1 != len2) {
            p2 = p2->next;
            len2--;
        }
    }
    while (p1!=NULL) {
        if (p1 == p2)
            return p1;
        else {
            p1 = p1->next;
            p2 = p2->next;
        }
    }
    return NULL;
}
#pragma endregion
#pragma region 53 - I. 在排序数组中查找数字 I
int search(vector<int>& nums, int target) {
    int count = 0;
    for (auto ele:nums) {
        if (ele == target)
            count++;
    }
    return count;
}
#pragma endregion

#pragma region II. 0～n-1中缺失的数字
int missingNumber(vector<int> & nums) {
    if (nums.size() == 0)
        return -1;

    int first = 0;
    int sec = nums.size() - 1;
    int mid = (first + sec) / 2;
    while (first != sec && sec - first > 1) {
        if (mid == nums[mid]) {
            first = mid;
            mid = mid = (first + sec) / 2;
        }
        else {
            sec = mid;
            mid = mid = (first + sec) / 2;
        }
    }
    if (first != nums[first])
        return first;
    if (sec != nums[sec])
        return sec;
    if (nums.size() - 1 == sec)
        return nums.size();
    return -1;
}
#pragma endregion

#pragma region 54. 二叉搜索树的第k大节点
int inOrderTree(TreeNode* root, int& k) {
    if (root == NULL)
        return -1;
    if (root->right != NULL) {
        int res = inOrderTree(root->right, k);
        if (res > -1) {
            return res;
        };
    }
    if (k == 1) {
        return root->val;
    }
    k--;
    if (root->left != NULL) {
        int res = inOrderTree(root->left, k);
        if (res > -1) {
            return res;
        };
    }
    return -1;
}
int kthLargest(TreeNode* root, int k) {
    return inOrderTree(root,k);
}
#pragma endregion

#pragma region I. 二叉树的深度
int maxDepth(TreeNode* root) {
    if (root == NULL)
        return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
}
#pragma endregion

#pragma region 55 - II. 平衡二叉树
#pragma region bad
// this computed depth multiple times
//bool isBalanced(TreeNode* root) {
//    if (root == NULL)
//        return true;
//    if (isBalanced(root->left) == false)
//        return false;
//    else if (isBalanced(root->right) == false) {
//        return false;
//    }
//    if (abs(maxDepth(root->left) - maxDepth(root->right)) > 1)
//        return false;
//    return true;
//}
#pragma endregion

// the sec is depth of the root
pair<bool, int> subIsBalanced(TreeNode* root) {
    if (root == NULL)
        return pair<bool, int>{true,0};
    pair<bool, int> res = subIsBalanced(root->left);
    bool isTrue = res.first;
    int depthL = res.second;
    if (isTrue == false)
        return pair<bool, int>{false, depthL};
    res = subIsBalanced(root->right);
    isTrue = res.first;
    int depthR = res.second;
    if (isTrue == false) {
        return pair<bool, int>{false, depthR};
    }
    if (abs(depthL - depthR) > 1)
        return pair<bool, int>{false, depthR};
    return pair<bool, int>{true, max(depthR,depthL)+1};
}

bool isBalanced(TreeNode* root) {
    return subIsBalanced(root).first;
}
#pragma endregion

#pragma region 57. 和为s的两个数字
vector<int> twoSum(vector<int>& nums, int target) {
    int p1 = 0;
    int p2 = nums.size() - 1;
    vector<int> res;
    while (p1!=p2) {
        if (nums[p2] > target) {
            p2--;
            continue;
        }
        if ((nums[p2] + nums[p1]) > target) {
            p2--;continue;
        };
        if ((nums[p2] + nums[p1]) < target) {
            p1++;continue;
        };
        if ((nums[p2] + nums[p1]) == target) {
            res.push_back(nums[p2]);
            res.push_back(nums[p1]);
            return res;
        }
    }
    return res;
}
#pragma endregion

#pragma region 57 - II. 和为s的连续正数序列
inline bool haveFloat(float num) {
    return (num - int(num)) != 0;
}

vector<vector<int>> findContinuousSequence(int target) {
    vector<vector<int>> res;
    float k = 100001;
    int i = 1;
    while (1) {
         k = (-(2*i-1) + sqrt((2 * i - 1)* (2 * i - 1) + 4 *2* float(target))) / 2;
         if (k <= 1)
             break;
        if (haveFloat(k)) {
            i++;
            continue;
        }
        vector<int> sub;int j = 0;
        while (j != k) {
            sub.push_back(i + j);j++;
        }
        i = i ++;
        res.push_back(sub);
    }
    return res;
}
#pragma endregion

#pragma region 58 - I. 翻转单词顺序
string reverseWords(string s) {
    if (s == "")
        return "";
    stringstream ss(s);
    string tmp;
    vector<string> strv;
    while (getline(ss, tmp, ' ')) {
        if (tmp.size() != 0)
            strv.push_back(tmp);
    }
    if (strv.size() == 0)
        return "";
    reverse(std::begin(strv), std::end(strv));
    tmp = "";
    while (strv.size() != 1) {
        tmp.append(strv[0]);
        tmp.append(" ");
        strv.erase(strv.begin());
    }
    tmp.append(strv[0]);
    return tmp;
}
#pragma endregion

#pragma region 58 - II. 左旋转字符串
string reverseLeftWords(string s, int n) {
    string tmp = s.substr(0,n);
    string tmp2 = s.substr(n,s.size()-n);
    tmp2.append(tmp);
    return tmp2;
}
#pragma endregion

#pragma region 59 - I. 滑动窗口的最大值
int help59[100000] = { 0 };
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    if (k <= 1)
        return nums;
    if (nums.size() <= k) {
        int ma = INT_MIN;
        int i = 0;
        while (i != nums.size()) {
            if (nums[i] > ma)
                ma = nums[i];
            i++;
        }
        return vector<int>{ma};
    }
    int head = 0;int tail = -1;int go = k-1;
    vector<int> res;
    for (int i = 0;i < k;i++) {
        if (i == 0) {
            help59[0]=(nums[0]);
            tail++;
        }
        else {
            while (tail>-1&&nums[i] > help59[tail]) {
                tail--;
            }
            tail++;
            tail = tail % 100000;
            help59[tail] = nums[i];
        }
    }
    res.push_back(help59[head]);
    for (go = k ;go < nums.size();go++) {
        if (nums[go-k] == help59[head]) {
            head++; head = head % 100000;
        }
        while (tail > head-1 && nums[go] > help59[tail]) {
            tail--;
        }
        tail++;
        tail = tail % 100000;
        help59[tail] = nums[go];
        res.push_back(help59[head]);
    }
    return res;
}
#pragma endregion

#pragma region 61. 扑克牌中的顺子
bool isStraight(vector<int>& nums) {
    for (int i = 0;i < nums.size();i++) {
        if (nums[i] == 'A')
            nums[i] = 1;
        else if(nums[i]=='J') {
            nums[i] = 11;
        }
        else if (nums[i] == 'Q') {
            nums[i] = 12;
        }
        else if (nums[i] == 'K') {
            nums[i] = 13;
        }
    }
    sort(nums.begin(),nums.end());// 如果使用hash，那么增加空间复杂度，降低时间复杂度
    int i = 0;int zeroCount = 0;int pre=-1;
    int mi = INT_MAX;int ma = INT_MIN;
    while (i!=nums.size()) {
        if(nums[i] == 0)
            zeroCount++;
        else if (pre == nums[i]) {
            return false;
        }
        pre = nums[i];i++;
    }
    mi = nums[zeroCount];ma = nums[nums.size() - 1];
    if (ma - mi > (nums.size() - 1 - zeroCount + zeroCount)) { return false; }
    return true;
}
#pragma endregion

#pragma region 62. 圆圈中最后剩下的数字
#pragma region bad
// take too many time
//int lastRemaining(int n, int m) {
//    vector<int> v;
//    int i = 0;
//    while (i!=n) {
//        v.push_back(i);
//        i++;
//    }
//    i = 1;
//    while (v.size()!=1) {
//        i = (i + (m-1))%v.size();
//        if (i == 0) {// last one
//            v.erase(v.end() - 1);
//            i = 1;
//        }
//        else {
//            v.erase(v.begin()+i-1);
//        }
//    }
//    return v[0];
//}
//
#pragma endregion

int lastRemaining(int n, int m) {
    if (n == 1) {
        return 0;
    }
    int f = 0;
    for (int i = 2;i <= n;i++) {
        f = (f + m) % i;
    }
    return f;
}
#pragma endregion

#pragma region 63. 股票的最大利润
//int subMaxProfit() {
//}
//int maxProfit(vector<int>& prices) {
//    int res = 0;
//    if (prices.size() <= 1)
//        return res;
//    for (int i = 0;i < prices.size();i++) {
//        while (i + 1<prices.size()&&prices[i]<prices[i+1]) {
//            i++;
//        }
//        for (int j = i;j < prices.size();j++) {
//            if (prices[j]>prices[i]) {
//                if (prices[j] - prices[i] > res)
//                    res = prices[j] - prices[i];
//            }
//        }
//    }
//    return res;
//}
int maxProfit(vector<int>& prices) {
    if (prices.size() < 1)
        return 0;
    if (prices.size() == 1)
        return 0;
    int mi = prices[0];
    int maProfit=0;
    for (int k = 0;k < prices.size();k++) {
        if (prices[k] < mi)             {
            mi = prices[k];
        }
        if (prices[k] - mi > maProfit)
            maProfit = prices[k] - mi;
    }
    return maProfit;
}
#pragma endregion


#pragma region 64. 求1+2++n
int sumNums(int n) {
    n&&(n = n + sumNums(n - 1));
    return n;
}
#pragma endregion


#pragma region 65. 不用加减乘除做加法
int add(int a, int b) {
    int mask = 1;
    //int stop = 0x80000000;
    int overflow = false;
    int sum = 0;
    while (mask) {
        int tmp1 = a & mask;
        int tmp2 = b & mask;
        if (overflow) {
            sum = mask ^ sum;
            if (mask == tmp2 || mask == tmp1)
                overflow = true;
            else {
                overflow = false;
            }
        }
        else {
            if (mask == tmp1 && mask == tmp2)
                overflow = true;
        }
        sum = sum ^ tmp1;
        sum = sum ^ tmp2;

        if (mask == 0x80000000)
            break;
        mask = mask << 1;

    }
    return sum;
}
#pragma endregion

#pragma region 66. 构建乘积数组
vector<int> constructArr(vector<int>& a) {
    if (a.size() == 0)
        return vector<int>();
    vector<int> first;int i = 0;
    while (i !=a.size()-1) {
        if (i == 0)
            first.push_back(a[0]);
        else {
            first.push_back(first[first.size()-1]*a[i]);
        }
        i++;
    }
    vector<int> secd;
    i = 0;
    while (i!=a.size()) {
        secd.push_back(0);i++;
    }
    i = a.size()-1;
    while (i>-1) {
        if (i == a.size() - 1)
            secd[i] = a[i];
        else {
            secd[i] = secd[i+1]*a[i];
        }
        i--;
    }
    vector<int> res;i = 0;
    while (i!=a.size()) {
        if(i==0)
            res.push_back(secd[1]);
        else if(i==a.size()-1)
            res.push_back(first[i-1]);
        else {
            res.push_back(first[i-1] * secd[i+1]);
        }
        i++;
    }
    return res;
}
#pragma endregion


#pragma region 68 - I. 二叉搜索树的最近公共祖先
TreeNode* lowestCommonAncestor1(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (root == NULL)
        return NULL;
    if (root->val > p->val && root->val < q->val || root->val<p->val && root->val > q->val)
        return root;
    if (root->val == p->val || root->val == q->val)
        return root;
    if (p->val < root->val && q->val < root->val)         {
        return lowestCommonAncestor1(root->left,p,q);
    }
    else {
        return lowestCommonAncestor1(root->right, p, q);
    }
    return NULL;
}
#pragma endregion

#pragma region 68 - II. 二叉树的最近公共祖先
pair<TreeNode*, int> subLowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q,int K) {
    // 第二个返回值表示在root为根的目录下找到了几个, 参数K表示需要找几个
    if (root == NULL) {
        return pair < TreeNode*, int>{NULL, 0};
    }
    if (K == 2) {
        if (root == p || root == q) {
            pair < TreeNode*, int> res = subLowestCommonAncestor(root->left, p, q, 1);
            if (res.second == 1)
                return pair < TreeNode*, int>{root, 2};//以root为根的子树找到2个
            if (res.second == 0) {
                pair < TreeNode*, int> res2= subLowestCommonAncestor(root->right, p, q, 1);
                if (res2.second == 1) {
                    return pair < TreeNode*, int>{root, 2};//以root为根的子树找到2个
                }
                else {
                    return pair < TreeNode*, int>{NULL, 1};//以root为根的子树只找到1个
                }
            }
        }
        pair < TreeNode*, int> res = subLowestCommonAncestor(root->left, p, q, 2);
        if (res.second == 2)
            return res;
        if (res.second == 1) {
            pair < TreeNode*, int> res= subLowestCommonAncestor(root->right, p, q, 1);
            if (res.second == 1)
                return pair < TreeNode*, int>{root,2};
            else {
                return pair < TreeNode*, int>{NULL, 1};
            }
        }
        if(res.second==0)
            return subLowestCommonAncestor(root->right, p, q, 2);
    }
    if (K == 1) {
        if (root == p || root == q) {
            return pair < TreeNode*, int>{NULL,1};
        }
        pair < TreeNode*, int> res = subLowestCommonAncestor(root->left, p, q, 1);
        if (res.second == 1)
            return res;
        if (res.second == 0)
            return subLowestCommonAncestor(root->right, p, q, 1);
    }
    
}

TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
     return subLowestCommonAncestor(root, p, q,2).first;
}
#pragma endregion



}
