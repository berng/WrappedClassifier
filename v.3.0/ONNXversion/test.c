#include <stdio.h>
#include <math.h>

#include "classifier_func.h"

int main()
{
 float inp[9]={1,1,1,1,1,1,1,1,1};
 float outp[35];
 float prob;
 ClassifierRun(inp,outp);
/// display
 for(int i=0;i<35;i++)
  printf("%e ",outp[i]);
 printf("\npredicted:%d\n",argmax(outp,&prob,35));
}

