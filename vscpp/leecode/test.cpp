#include <iostream>
#include <string>
#include <ctime>
#include <array>
#include <vector>
#include <queue>
#include <list>
#include <map>
#include <stack>

#include <numeric>
#include <utility>

#include <algorithm>
using namespace std;

template <typename Container>
void print(const Container& container) {
    for (const auto& element : container) {
        cout << element << ' ';
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


vector<int> asteroidCollision(vector<int>& asteroids) {
    int index_max=0;
    for(int i=0;i<asteroids.size();i++){
        if(abs(asteroids[i])>abs(asteroids[index_max])){
            index_max = i;
        }
    }
    if(asteroids[index_max]<0){
        std::vector<int> result(asteroids.size());
        transform(asteroids.begin(), asteroids.end(), result.begin(), [](int num) { return -num; });
        reverse(asteroids.begin(), asteroids.end());
    }
}

int main(){
    vector<vector<int>> matrix = {{1,3},{2,4},{3,4}};
    vector<int> plains = {10,2,-5};
    vector<int> a = asteroidCollision(plains);
    print(a);
    // cout << a << endl;
}






