#include<iostream>
#include <vector>
using namespace std;
int main()
{

    vector <int> test_scores {10,20,30,40};

    cin >> test_scores.at(0);
    cin>> test_scores.at(1);

    test_scores.at(0)=90;

    test_scores.push_back(80);
    cout<< test_scores[test_scores.size()-1];
}