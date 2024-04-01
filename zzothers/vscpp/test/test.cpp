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


struct A{
public:
    int a = 1;
    int* p;
    A() : p(&a) {}
};

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

void testp(){
    int a = 1;

    vector<int> l = {1,2,3};

    int* p;
    p = &a;

    A ca = A();
    A* pca = &ca;
    TreeNode node_sub = TreeNode(2);
    TreeNode* pnode_sub = &node_sub;
    TreeNode node = TreeNode(1,pnode_sub,pnode_sub);
    TreeNode* pnode = &node;
    cout << pca->a << endl;
    cout << pnode->val << endl;
};

int main() {
    char a = 'a';
    int b = 2;
    cout << int(a);
}









