#include <iostream>
#include <string>
#include <ctime>
#include <array>
#include <vector>
#include <queue>
#include <list>
#include <map>

#include <algorithm>
using namespace std;

template <typename Container>
void print(const Container& container) {
    for (const auto& element : container) {
        cout << element << " ";
    }
    cout << endl;
}
template <typename Container>
void print2d(const Container& container) {
    for (const auto& element : container) {
        print(element);
    }
    cout << endl;
}


void testvec(){
    vector<vector<int>> matrix = {{3,2},{1,4},{5,6}};
    sort(matrix.begin(),matrix.end());
    for(vector<int> vec : matrix){
        print(vec);
    }
}

int main() {
    vector<vector<int>> matrix = {{3,2},{1,4},{5,6}};
    vector<int> vec = {1,2,3,4,5};
    vector<int>::iterator a = find(vec.begin(),vec.end(),3);
    cout << *vec.begin() << endl;
    

}









