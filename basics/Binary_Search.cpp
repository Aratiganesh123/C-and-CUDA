#include<iostream>
#include<climits>
using namespace std;

int BinarySearch(int arr[],int n, int flag)
{

int s=0;
int e=n;

while(s<=e)
{
    int mid=(s+e)/2;
    if(arr[mid]==flag)
    {
        return mid;
    }
    else if(arr[mid]>flag){
        e=mid-1;
    }
    else{
        s=mid+1;
    }
}
}

int main(){

int n;
cout<<"Enter n :"; 
cin>>n;
int arr[n];

for(int i=0;i<n;i++)
{
    cin>>arr[i];
}

int flag;
cout<< "Enter Flag :";
cin>>flag;

cout<<"Position of Key is  "<<BinarySearch(arr,n,flag)<<endl;

}



