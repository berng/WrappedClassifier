#include <stdio.h>
#include <math.h>

#include "classifierV3.h"

int main()
{
 float inp[9]={1,1,1,1,1,1,1,1,1};
 float outp[35];
 float prob;
 predictV3(inp,outp);
/// display
 for(int i=0;i<35;i++)
  printf("%e ",outp[i]);
 printf("\npredicted:%d\n",argmax(outp,&prob,35));
}

