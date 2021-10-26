#include<iostream>
#include<string>
#include<vector>

using namespace std;

int main()
{
    int num{100};
    int &ref{num};

    num=300;
    cout<<num<< " "<<ref;

    vector<string> stooges{"a","b","c"};
    for(auto str:stooges)
       str="Funny";
     for(auto str:stooges)
        cout<<str<<" ";
    for(auto &str:stooges)
       str="Funny";
     for(auto const &str:stooges)
        cout<<str<<" ";

    
}