#include<iostream>
#include<string>
#include<vector>

using namespace std;

using namespace std; 

int* largest_int(int *ptr1,int* ptr2)
{
    if(*ptr1>*ptr2)
        return ptr1;
    else
        return ptr2;
    

}

int main()
{
    int a{100};
    int b{200};

    int* largest_value{nullptr};
    largest_value=largest_int(&a,&b);
    cout<<"Largest value : "<<*largest_value<<endl;

}