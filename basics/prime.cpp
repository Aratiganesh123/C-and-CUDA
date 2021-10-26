#include <iostream>
#include<cmath>
using namespace std;

int main()
{
    int number;
    cout<<"Enter a number";
    cin>>number;
    int i,flag=0;
    for(i=2;i<sqrt(number);i++)
    {
        if(number%i==0)
        {
            cout<<"This number is not Prime";
            break;
            flag=1;
        }
    }
printf("%d",flag);
   if(flag==0){
    cout<<"This number is  Prime";
   }
   
}