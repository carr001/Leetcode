#include<iostream>
#include<sstream>
#include<vector>
#include<algorithm>
#include<string>
using namespace std;
namespace newcoder {

int oj_test1() {
	int a;int b;
	while (cin >> a >> b) {
		cout << a + b << endl;
	};
	return 0;
}
int oj_test2() {
	int t;
	cin >> t;
	while (t--) {
		int a;int b;
		cin >> a >> b;
		cout << a + b << endl;
	}
	return 0;
}
int oj_test3() {
	int a;int b;
	while (cin>>a>>b) {
		if (a == 0 && b == 0) {
			break;
		}
		else {
			cout << a + b << endl;
		}
	}
	return 0;
}
int oj_test4() {
	int a;
	while (cin >> a) {
		if (a == 0)
			break;
		int b;int sum = 0;
		while (a--) {
			cin >> b;
			sum += b;
		}
		cout << sum << endl;
	}
	return 0;
}
int oj_test5() {
	int a;
	cin >> a;
	while (a--) {
		int b;int c;int sum = 0;
		cin >> b;
		while (b--) {
			cin >> c;
			sum += c;
		}
		cout << sum << endl;
	}
	return 0;
}
int oj_test6() {
	int a;
	while (cin >> a) {
		int b;int sum = 0;
		while (a--) {
			cin >> b;
			sum += b;
		}
		cout << sum << endl;
	}
	return 0;
}
int oj_test7() {
	int nums;
	int sum = 0;
	while (cin >> nums) {
		sum += nums;
		if (cin.get() == '\n') {
			cout << sum << endl;
			sum = 0;
		}
	}
	return 0;
}
int oj_test8() {
	int n;
	string s;
	vector<string> ss;
	cin >> n;
	while (n--) {
		cin >> s;
		ss.push_back(s);
	};
	sort(ss.begin(), ss.end());
	for (int i = 0;i < ss.size();i++) {
		cout << ss[i] << ' ';
	}
	return 0;
}
int oj_test9() {
	vector<string> ss;
	string s;
	while (cin >> s) {
		ss.push_back(s);
		if (cin.get() == '\n') {
			sort(ss.begin(), ss.end());
			for (int i = 0;i < ss.size();i++) {
				cout << ss[i] << ' ';
			}
			cout << endl;
			ss.clear();
		}
	}
	return 0;
}
int oj_test10() {
	string line, tmp;
	while (getline(cin, line)) {
		stringstream ss(line);
		vector<string> strv;
		while (getline(ss, tmp, ',')) {
			strv.push_back(tmp);
		}
		sort(strv.begin(), strv.end());
		for (int i = 0;i < strv.size() - 1;i++) {
			cout << strv[i] << ',';
		}
		cout << strv.back() << endl;
	}
	return 0;
}
int oj_test11() {
	long long a;long long b;
	while (cin >> a >> b) {
		cout << a + b << endl;
	}
	return 0;
}



}
