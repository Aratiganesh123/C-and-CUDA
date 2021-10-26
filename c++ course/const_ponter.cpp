#include<iostream>
#include<string>
#include<vector>

using namespace std;
void display(vector<string>* v)
{
    for(auto str:*v)
        cout<<str<<" ";
    cout<<endl;
    (*v).at(0)="I've changed";
    for(auto str:*v)
        cout<<str<<" ";
    cout<<endl;
}
void display(int* array,int sentinel)
{
    while(*array !=sentinel)
        cout<<*array++<<" ";
       
    cout<<endl;
    
}


int main()
{
    // vector<string> stooges{"a","b","c"};
    // display(&stooges);

    int scores[]{100,98,97,-1};
    display(scores,-1);
}