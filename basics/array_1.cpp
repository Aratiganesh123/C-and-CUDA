#include<iostream>
#include<climits>
using namespace std;

int main(){

int n;
cin>>n;
int arr[n];

for(int i=0;i<n;i++)
{
    cin>>arr[i];
}
for(int i=0;i<n;i++)
{
    cout<<arr[i]<<endl;
}

int maxNum=INT_MIN;
int minNUM=INT_MAX;

for(int i=0;i<n;i++){
   maxNum=max(maxNum,arr[i]);
   minNUM=min(minNUM,arr[i]);
}

cout<<"MIN : "<<minNUM<<endl;
cout<<"MAX : "<<maxNum<<endl;


}