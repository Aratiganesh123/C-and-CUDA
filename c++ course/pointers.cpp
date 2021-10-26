#include<iostream>
#include<vector>
#include<string>

using namespace std;

int main()
{
    int *p;

    cout<< "Value of p is : " << p <<endl;
     cout<< "address of p is : " << &p <<endl;
     cout<<" Size of p is : "<< sizeof(p)<<endl;
     p=nullptr;
     cout<<" P initialised to 0"<<p;

     int score{100};
    int* score_ptr{&score};

    cout<<*score_ptr<<endl;

    *score_ptr=200;
    cout<<score<<endl;

    vector<string> stooges{"pop","lop","kop"};
    vector<string>* vector_ptr{nullptr};
    vector_ptr=&stooges;

    cout<<"First Stooge :"<< (*vector_ptr)[0]<<endl;

    for(auto stooge:*vector_ptr)
        cout<<stooge<<" ";
    cout<<endl;


}