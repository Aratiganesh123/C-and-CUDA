#include<iostream>

using namespace std;

int main()
{
    int scores[] {10,22,44,55};

    int* score_ptr{scores};

    // while(*score_ptr!=55)
    // {
    //     cout<<*score_ptr<<endl;
    //     score_ptr++;
    // }
     while(*score_ptr!=55)
    {
        cout<<*score_ptr++<<endl;
       
    }
}