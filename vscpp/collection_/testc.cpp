#include <iostream>
#include <string>
#include <ctime>
#include <array>
#include <list>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <numeric>

using namespace std;

// array（数组）：
// 固定大小的容器，其大小在创建时确定且无法改变。
// 元素在内存中是连续存储的。
// 可以使用下标访问元素，时间复杂度为O(1)。
// 不支持动态插入或删除元素。
// 头文件：<array>

// list（链表）：
// 动态大小的容器，可以动态添加或删除元素。
// 元素在内存中不一定是连续存储的，而是通过指针链接在一起。
// 访问元素需要遍历链表，时间复杂度为O(n)。
// 支持高效的插入和删除操作，时间复杂度为O(1)。
// 头文件：<list>

// vector（动态数组）：
// 动态大小的容器，可以动态添加或删除元素。
// 元素在内存中是连续存储的。
// 可以使用下标访问元素，时间复杂度为O(1)。
// 支持高效的尾部插入和删除操作，时间复杂度为O(1)。
// 在需要频繁访问元素或在尾部进行插入和删除操作时效率较高。
// 头文件：<vector>
void test(){
    array<int, 5> arr1 = {1, 2, 3, 4, 5};
    cout << arr1 << endl;
}

int main(){
    test();
}

// ======================array======================
void testarr() {
    array<int, 5> arr1 = {1, 2, 3, 4, 5};

    arr1[3] = 10;
    for (int num : arr1) {
        cout << num << " ";
    }

}

// ======================list=======================
void testlist() {
    list<int> myList;

    myList.push_back(10);
    myList.push_front(5);
    myList.pop_front();
    for (int num : myList) {
        cout << num << " ";
    }
    for (std::list<int>::iterator it = myList.begin(); it != myList.end(); ++it) {
        std::cout << *it << " ";
    }
}

// ======================vecor======================
void testvec(){
    //创建
    vector<int> vec1 = {1, 2, 3, 4, 5};
    vector<int> vec2(10);     // 创建包含10个元素的向量，每个元素都被初始化为0
    vector<int> vec3(5, 10);  // 创建包含5 个元素的向量，每个元素都被初始化为10

    vector<int> vec4;
    array<int, 5> arr = {1, 2, 3, 4, 5};
    vec4.assign(arr.begin(), arr.end());

    vector<int> originalVec = {1, 2, 3};
    vector<int> copiedVec(originalVec);  // 通过复制构造函数创建一个与原向量相同的副本

    //创建二维vector
    vector<vector<int>> matrix = {{1,2},{3,4},{5,6}};
    vector<std::vector<int>> matrix(2,vector<int>(3, 0));
    

    //常用操作
    vector<int> vec;
    vec.push_back(10);
    vec[0] = 25;
    vec.insert(vec.begin() + 2, 15);
    vec.pop_back();
    for (int num : vec) {
        cout << num << " ";
    }

    int sum = accumulate(vec.begin(), vec.end(), 0);

}

// ======================map======================
void testmap(){
    map<int, string> myMap = {{1, "apple"}, {2, "banana"}, {3, "orange"}};
    myMap[4] = "pear";
    for (const auto& pair : myMap) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
}

// ======================set======================

void testset(){
    set<int> mySet = {1, 2, 3, 4, 5};
    mySet.insert(6);
    for (const auto& value : mySet) {
        std::cout << value << " ";
    }
}

// ======================queue======================
void testqueue(){
    queue<int> q1;
}