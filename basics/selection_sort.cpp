/*
Selection Sort

12,45,23,51,19,8
find 1st minimum element - 8,45,23,51,19,12
2nd - 8,12,23,51,19,45
3rd - 8,12,19,51,23,45

*/

#include<iostream>

using namespace std;

int main()
{

    int n;
    cin>>n;
    int arr[n];

    for(int i=0;i<n;i++)
    {
        cin>>arr[i];
    }

    for(int i=0;i<n-1;i++)
    {
        for(int j=i+1;j<n;j++)
        {
            if(arr[j]<arr[i])
            {
                int temp=arr[j];
                arr[j]=arr[i];
                arr[i]=temp;
            }
        }
    }
    for(int i=0;i<n;i++)
{
    cout<<arr[i]<<endl;
}
}

