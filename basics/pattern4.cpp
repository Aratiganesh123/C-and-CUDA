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
{       int b=i;
       int a=2;
       
    count=1;
    for(j=1;j<=(n-i);j++)
    {
        cout<<" ";
    }
    for(k=1;k<=(2*i-1);k++)
    {
       
   
       if(k<=i)
       {   
           cout<<b;
           cout<<" ";
           b=b-1;
       }
       else if(k>i)
       {
           cout<<a;
           cout<<" ";
           a=a+1;
       }

    }
cout<<endl;
}


}