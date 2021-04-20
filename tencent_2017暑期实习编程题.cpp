#include <iostream>
#include<algorithm>
#include<vector>
using namespace std;
int memo[1001][1001] = { 0 };// 静态数组尽量声明成全局的
int maxPar(string ss,int p1,int p2) {
    if (p1 > p2)
        return 0;
    if (p1 == p2)
        return 1;
    if (memo[p1][p2] > 0)
        return memo[p1][p2];
    int maxLen = 0;
    if (ss[p1] == ss[p2]) {
        maxLen = maxPar(ss, p1+1,p2-1)+2;
    }
    else {
        int tmp = maxPar(ss, p1, p2 - 1);
        int tmp2 = maxPar(ss, p1+1, p2);
        maxLen = max(tmp,tmp2);
    }
    memo[p1][p2] = maxLen;
    return maxLen;
}
int quiz_1() {
    string ss;
    while (cin >> ss) {
        int len = ss.length();
        int p1 = 0, p2 = len;
        /*int** memo = (int**)malloc(sizeof(int*) * len);// 尽量少用，这样总是出错，原因未知
        for (int i = 0;i < len;i++) {
            memo[i] = (int*)malloc(sizeof(int) * len);
            for (int j = 0;j < len;j++) {
                memo[i][j] = 0;
            }
        }*/
        for (int i = 0;i < 1001;i++) {// 每次必须要将所有的内存设为0
            for (int j = 0;j < 1001;j++) {
                memo[i][j] = 0;
            }
        }
        int maxLen = maxPar(ss, p1, p2);
        cout << len - maxLen << endl;
        ss.clear();
    }
    return 0;
}
int mswap(char* p1, char*p2) {
    char tmp = *p1;
    *p1 = *p2;*p2 = tmp;
    return 0;
}
int quiz_2() {
    string s;
    while (cin >> s) {
        char* p2 = &s[s.size() - 1];
        char* p1 = &s[0];
        for (char* p11 = &s[0];p11 != p2;p11++) {
            if (*p11 >= 'a' && *p11 <= 'z') {
                char* tmp1 = p11;
                while (tmp1 != p1) {
                    mswap(tmp1, tmp1 - 1);
                    tmp1--;
                }
                p1++;
            }
        }
        if (*p2 >= 'a' && *p2 <= 'z') {
            char* tmp1 = p2;
            while (tmp1 != p1) {
                mswap(tmp1, tmp1 - 1);
                tmp1--;
            }
            p1++;
        }
        cout << s << endl;
        s.clear();
    }
    return 0;
}





