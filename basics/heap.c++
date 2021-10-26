#include<iostream>
using namespace std;

int main()
{
    int a=10; // in stack
    int *p= new int(); // in heap

    *p=10; // manipulate it from the stack
    delete(p);

    p=new int[4];

    delete[]p;

    p=NULL;

// memory leak happens when you allocate memory in heap and you do not delete the memory
// therefore free the memory allocated dynamically
}