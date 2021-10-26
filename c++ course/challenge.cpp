#include<iostream>
#include<string>
#include<vector>

using namespace std;

int* create_new(int* arr1,int* arr2, size_t size1,size_t size2)
{
    int* new_array{nullptr};

    new_array= new int[size1*size2];

    int count{0};

    for(size_t i{0};i<size1;i++)
    {
        for(size_t j{0};j<size2;j++)
        {
            *(new_array+count) = arr1[i]*arr2[j];
            count++;
        }
    }
    return new_array;
}

void display(int* results,size_t size)
{
    for(size_t i{0};i<size;i++)
    {
        cout<<results[i]<< " "<< i<<endl;
    }
}

int main()
{
    int arr1[]{1,2,3,4,5};
    int arr2[]{6,7,8};

    int* results{nullptr};

    results=create_new(arr1,arr2,5,3);

    display(results,15);

    delete[] results;



}