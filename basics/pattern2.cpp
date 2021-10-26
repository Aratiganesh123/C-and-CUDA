#include<iostream>
using namespace std; 
int main()
{
int n;
printf("enter a number");
cin>>n;
int i,j;
for(i=1;i<=n;i++)
{
    for(j=1;j<=(2*n -i);j++)
    {
        if(j<=(n-i))
        {
            cout<<" ";

        }
        else{
            cout<<"*";
        }
    }
    cout<<endl;
}
}