#include <iostream>
#include <string>
#include <ctime>
#include <array>
#include <vector>
#include <queue>
#include <list>
#include <map>
#include <unordered_map>
#include <stack>

#include <sstream>

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


class Solution {
public:
    int romanToInt(string s) {
        unordered_map<char, int> map = {{'I', 1}, {'V', 5}, {'X', 10},{'L', 50}, {'C', 100},{'D', 500}, {'M', 1000}};
        int ans = 0;
        int i = 1;
        while(i<s.size()){
            if(map[s[i-1]]>=map[s[i]]){
                ans += map[s[i-1]];
            }else{
                ans -= map[s[i-1]];
            };
            i += 1;
        };
        ans += map[s[i-1]];
        return ans;
    }
};

int main(){
    Solution slt = Solution();
    string s = "III";
    int result = slt.romanToInt(s);

    cout << result << endl;
}





