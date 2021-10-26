#include<iostream>
using namespace std;
int main()
{
#if 0
int n;
printf("enter a number");
cin>>n;
int i,j;
int count=1;
for(i=n;i>=1;i--)
{
   
    for(j=1;j<=i;j++)
    {
        cout<<count;
        count++;

    }
    cout<<"\n";
count=1;
}

int n;
printf("enter a number");
cin>>n;
int i,j;
int count=1;
for(i=1;i<=n;i++)
{
   
    for(j=1;j<=i;j++)
    {
        if(i%2==0)
        {
            if(j%2==0){
                cout<<"0";
            }
            else{
                cout<<"1";
            }

        }
      else{
           if(j%2==0){
                cout<<"1";
            }
            else{
                cout<<"0";
            }

      }

    }
    cout<<"\n";
count=1;
}


int n;
printf("enter a number");
cin>>n;
int i,j;
int count=1;
for(i=1;i<=n;i++)
{
   
    for(j=1;j<=i;j++)
    {   
        if(i%2==0)
        {
           cout<< ~count;
        }
      else{
           if(j%2==0){
                cout<<"1";
            }
            else{
                cout<<"0";
            }

      }

    }
    cout<<"\n";
count=1;
}

return 0;
#endif  



}