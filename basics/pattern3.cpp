#include<iostream>
using namespace std; 
int main()
{
int n;
printf("enter a number");
cin>>n;
int i,j,k;
int count;
for(i=1;i<=n;i++)
{
    count=1;
    for(j=1;j<=(n-i);j++)
    {
        cout<<" ";
    }
    for(k=1;k<=i;k++)
    {
        cout<<count;
        cout<<" ";
        count++;
    }
cout<<endl;
}


}