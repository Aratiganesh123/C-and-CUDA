#include<iostream>
#include<vector>
#include<string>

using namespace std;

//use new keyword to allocate storage at runtime
int main()
{

int *int_ptr {nullptr};

int_ptr = new int; // stores the address of int location on the heap

cout<<*int_ptr<< endl;
*int_ptr=100;

cout<<*int_ptr<<endl;

delete int_ptr;


int* array_ptr{nullptr};
int size{};

cout<<"What should the array size be :";
cin>>size;

array_ptr= new int[size];

(*array_ptr)=100;


    // for(auto stooge:*array_ptr)
    //     cout<<stooge<<" ";
    // cout<<endl;


cout<< *array_ptr[1]<<endl;




}