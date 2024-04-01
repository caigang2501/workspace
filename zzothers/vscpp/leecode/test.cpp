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
    int searchInsert(vector<int>& nums, int target) {
        for (int i = 0; i < nums.size(); ++i) {
            if (target<=nums[i]){
                return i;
            }
        }
        return -1;
    }
};

int main(){
    Solution s = Solution();
    vector<int> lst = {1,2,4,5};
    int result = s.searchInsert(lst,3);
    cout << result << endl;
}





