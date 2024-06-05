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
    string gcdOfStrings(string str1, string str2) {
        vector<int> max_lens(str2.size()+1,0);
        vector<int> max_lens_copy = max_lens;
        int max_ = 0;
        int index_ = 0;
        for(char c1:str1){
            max_lens_copy = max_lens;
            for (int i = 0;i<str2.size();i++) {
                if(c1==str2[i]){
                    max_lens[i+1] = max_lens_copy[i]+1;
                    if(max_lens[i+1]>max_){
                        max_ = max_lens[i+1];
                        index_ = i;
                    }
                }else{
                    max_lens[i+1] = 0;
                }
            }
        }
        cout << max_ << index_ << endl;
        return str2.substr(index_-max_+1,max_);
    }
};


int main(){
    Solution s = Solution();
    string a = s.gcdOfStrings("asdf","asdfasdf");
    string str = "asdfasdf";
    cout << a << endl;
}





