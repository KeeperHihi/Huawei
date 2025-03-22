#include <bits/stdc++.h>
using namespace std;

#define MAX_TAG 16
#define MAX_EPOCH 86506

int tag_frequency[MAX_TAG + 1][MAX_EPOCH];

int main() {
	ifstream fin("./CppCode/tag_frequency.txt");
	for (int i = 1; i <= MAX_TAG; i++) {
		for (int j = 0; j < MAX_EPOCH; j++) {
			fin >> tag_frequency[i][j];
		}
	}
	for (int i = 1; i <= MAX_TAG; i++) {
		cerr << accumulate(tag_frequency[i], tag_frequency[i] + MAX_EPOCH, 0LL) << ' ';
	} cerr << endl;
	return 0;
}