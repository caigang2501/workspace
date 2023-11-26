#include <boost/shared_ptr.hpp>
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

int main() {
    boost::shared_ptr<int> sharedInt(new int(42));
    std::cout << "Value: " << *sharedInt << std::endl;
    std::cout << "Use count: " << sharedInt.use_count() << std::endl;

    // sharedInt will be automatically destroyed when not needed
    return 0;
}

// int main(){
//     cout << "Use count: " << endl;
// }