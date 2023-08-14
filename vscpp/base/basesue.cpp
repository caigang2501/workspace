#include <iostream>
#include <string>
#include <ctime>
#include <vector>
#include <queue>
using namespace std;

void memery(){
    int a = 1;
    vector<int> vec;
    cout << typeid(a).name() << endl;
    vec.erase(vec.begin() + 2);
    priority_queue<int> pq;
    string str;
    vector<char> vec(str.begin(), str.end());
}

void test1(){
    auto name = "Lady G.";
}

void test2(){
    int a = 11;
    int *b = &a;
    int c = a;
    a = 12;
    
    cout << "a :" << endl;
    cout << "&a:" << &a << endl;
    cout << "b :" << b << endl;
    cout << "*b:" << *b << endl;
    cout << "&b:" << &b << endl;
    cout << "c :" << c << endl;
    cout << "&c:" << &c << endl;
}

void fun1(int &a){
    a += 1;
}

void test3(){
    int a = 1;
    fun1(a);
    cout << a << endl;
}

int main(){
    // test2();
    test2();
}


