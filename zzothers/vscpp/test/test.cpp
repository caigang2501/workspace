#include <iostream>
#include <string>
#include <ctime>
#include <array>
#include <vector>
#include <queue>
#include <deque>
#include <list>
#include <map>

#include <algorithm>
using namespace std;

template <typename Container>
void print_l(const Container& container) {
    for (const auto& element : container) {
        cout << element << " ";
    }
    cout << endl;
}

template <typename Container>
void print_2dl(const Container& container) {
    for (const auto& element : container) {
        print(element);
    }
    cout << endl;
}


void testvec(){
    vector<vector<int>> matrix = {{3,2},{1,4},{5,6}};
    sort(matrix.begin(),matrix.end());
    for(vector<int> vec : matrix){
        print_l(vec);
    }
}



int main() {
    deque<int> a;
    string s = "asdf";
    cout << s[2];
    for(char c:s){
        char a = s[1];
    }
}









