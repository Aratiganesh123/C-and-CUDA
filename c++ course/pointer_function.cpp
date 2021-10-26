#include<iostream>
#include<string>
#include<vector>

using namespace std;


int* create_array(size_t size , int init_value=0)
{
    int* new_storage{nullptr};
    new_storage = new int[size];
    for(size_t i{0};i<size;i++)
        *(new_storage+i)=init_value;
    return new_storage;
}
void display(int* array,size_t size)
{
    for(size_t i{0};i<size;i++)
    {
        cout<< array[i]<<" ";
    }
}


int main()
{
    int *my_array{nullptr};
    size_t size;
    int init_value{1};

    my_array=create_array(6,4);
    display(my_array,6);
    delete [] my_array;


}