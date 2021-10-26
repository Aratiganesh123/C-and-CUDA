#include<iostream>
#include<cmath>

using namespace std;


bool checkpytrip(int x,int y,int z)
{
int a=max(x,max(z,y));
int b,c;
if(a==x){
    b=y;
    c=z;
}
else if(a==y){
    b=x;
    c=z;

}
else{
    b=x;
    c=y;
}
if(a*a==b*b+c*c)
{
    return 1;
}
else{
    return 0;
}
}
int main(){

int x,y,z;
cin>>x>>y>>z;
if(checkpytrip(x,y,z)==1)
{
    cout<<"Is a pythogoream triplet";

}
else{
    cout<<"Not a triplet";
}
}