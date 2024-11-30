#include "classifierV3.h"
#include "classifierV3_coefs0.h"
#include "classifierV3_coefs1.h"
#include "classifierV3_coefs2.h"


/// Class predictor, V3
int predictV3_0(float* inp, float*outp)
/// inp:
/// V,Wl,Hiri,Mode,sin(k,xy)[R],cos(k,B)[R],sin(k,xy)[R/4],sin(k,xy)[R/2],sin(k,xy)[3R/4]
/// outp: class probabilities (Kolmogorov-like)
{
 double x[100];
 double y[100];
 for(int i=0;i<100;i++)
  x[i]=y[i]=0.;

/// batch normalization
 for(int i=0;i<9;i++)
  x[i]=(inp[i]-weights_bn_2_0[i])/sqrt(weights_bn_3_0[i]+1e-3);
 for(int i=0;i<9;i++)
   x[i]=x[i]*weights_bn_0_0[i]+weights_bn_1_0[i];

/// transform1
 for(int i=0;i<49;i++)
  {
   y[i]=weights_tr1_1_0[i];
   for(int j=0;j<9;j++)
    y[i]+=weights_tr1_0_0[j][i]*x[j];
  }
 for(int i=0;i<49;i++)
  x[i]=fabs(y[i]);

/// transform2
 for(int i=0;i<35;i++)
  {
   y[i]=weights_tr2_1_0[i];
   for(int j=0;j<49;j++)
    y[i]+=weights_tr2_0_0[j][i]*x[j];
  }
 for(int i=0;i<35;i++)
   x[i]=fabs(y[i]);

/// norm layer
 double sum=0;
 for(int i=0;i<35;i++)
  sum+=x[i];
 for(int i=0;i<35;i++)
  outp[i]=x[i]/sum;
 float* prob;
 return argmax(outp,&prob,35);
}

int predictV3_1(float* inp, float*outp)
/// inp:
/// V,Wl,Hiri,Mode,sin(k,xy)[R],cos(k,B)[R],sin(k,xy)[R/4],sin(k,xy)[R/2],sin(k,xy)[3R/4]
/// outp: class probabilities (Kolmogorov-like)
{
 double x[100];
 double y[100];
 for(int i=0;i<100;i++)
  x[i]=y[i]=0.;

/// batch normalization
 for(int i=0;i<9;i++)
  x[i]=(inp[i]-weights_bn_2_1[i])/sqrt(weights_bn_3_1[i]+1e-3);
 for(int i=0;i<9;i++)
   x[i]=x[i]*weights_bn_0_1[i]+weights_bn_1_1[i];

/// transform1
 for(int i=0;i<49;i++)
  {
   y[i]=weights_tr1_1_1[i];
   for(int j=0;j<9;j++)
    y[i]+=weights_tr1_0_1[j][i]*x[j];
  }
 for(int i=0;i<49;i++)
  x[i]=fabs(y[i]);

/// transform2
 for(int i=0;i<35;i++)
  {
   y[i]=weights_tr2_1_1[i];
   for(int j=0;j<49;j++)
    y[i]+=weights_tr2_0_1[j][i]*x[j];
  }
 for(int i=0;i<35;i++)
   x[i]=fabs(y[i]);

/// norm layer
 double sum=0;
 for(int i=0;i<35;i++)
  sum+=x[i];
 for(int i=0;i<35;i++)
  outp[i]=x[i]/sum;
 float* prob;
 return argmax(outp,&prob,35);
}

int predictV3_2(float* inp, float*outp)
/// inp:
/// V,Wl,Hiri,Mode,sin(k,xy)[R],cos(k,B)[R],sin(k,xy)[R/4],sin(k,xy)[R/2],sin(k,xy)[3R/4]
/// outp: class probabilities (Kolmogorov-like)
{
 double x[100];
 double y[100];
 for(int i=0;i<100;i++)
  x[i]=y[i]=0.;

/// batch normalization
 for(int i=0;i<9;i++)
  x[i]=(inp[i]-weights_bn_2_2[i])/sqrt(weights_bn_3_2[i]+1e-3);
 for(int i=0;i<9;i++)
   x[i]=x[i]*weights_bn_0_2[i]+weights_bn_1_2[i];

/// transform1
 for(int i=0;i<49;i++)
  {
   y[i]=weights_tr1_1_2[i];
   for(int j=0;j<9;j++)
    y[i]+=weights_tr1_0_2[j][i]*x[j];
  }
 for(int i=0;i<49;i++)
  x[i]=fabs(y[i]);

/// transform2
 for(int i=0;i<35;i++)
  {
   y[i]=weights_tr2_1_2[i];
   for(int j=0;j<49;j++)
    y[i]+=weights_tr2_0_2[j][i]*x[j];
  }
 for(int i=0;i<35;i++)
   x[i]=fabs(y[i]);

/// norm layer
 double sum=0;
 for(int i=0;i<35;i++)
  sum+=x[i];
 for(int i=0;i<35;i++)
  outp[i]=x[i]/sum;
 float* prob;
 return argmax(outp,&prob,35);
}




int argmax(float* arr, float* prob, int width_out_)
 {
  int pos=-1;
  float max=-1;
  for(int i=0;i<width_out_;i++)
   if (arr[i]>max)
    { max=arr[i]; pos=i; }
  if (prob)
   *prob=max;
  return pos;
 }

// ensemble realization
int predictV3(float* inp, float*outp)
{
 int res0=predictV3_0(inp, outp);
 int res1=predictV3_1(inp, outp);
 int res2=predictV3_2(inp, outp);
 if (res0==res1 && res0==res2)
  return res0;
 else
  return 35;
}


#ifdef __TEST__
/*
int main()
{
 double inp[9]={1,1,1,1,1,1,1,1,1};
 double outp[35];
 predictV3(inp,outp);
/// display
 for(int i=0;i<35;i++)
  printf("%e ",outp[i]);
 printf("\n");
}
*/
#endif

