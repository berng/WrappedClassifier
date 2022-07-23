#/usr/bin/python3
import sys
#RADAR='magw'
FILENAME=sys.argv[1]
MODEL=sys.argv[2]
OUTFILENAME=sys.argv[3]
MODE="withHeff"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

#output_class_M=46
#datesNum=199

DETECTION_ACCURACY=0.5

#output_class_M=31
#datesNum=212
vecInputNum=7+7
totalClassNum=25      
hidden_dim=32

#f=open("out."+RADAR+".nn","rt")
#f=open("out."+RADAR+".test","rt")
f=open(FILENAME,"rt")
lines=f.read().split("\n")
XARR1 = []
XARR1_INP = []
YARR1 = []
ASYMMETRY=dict()
i = 0
classes_sel=dict()
count_classes=0
Cl_Y=[]

print ("load model")
model=tf.keras.models.load_model(MODEL)
model.compile()
print(model)
model.summary()
print ("presave model")
keras.utils.plot_model(model, "encoder_model.png", show_shapes=True)

print ("check layer with hidden classes:")

LAYER_OF_SECONDARY_VARS=7

stat=dict()


# MAX_TRAIN_SIZE=1000000
MAX_TRAIN_SIZE=180000000
#MAX_BLOCK_SIZE=100000
MAX_BLOCK_SIZE=1
fout=open(OUTFILENAME,'w')
XARR1=[]
YARR1=[]
DATES1=[]

from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(2)
indexes2sel=[0,4,1,8,   6,7,9,10,11,12,13,14,15,16, 17,18]
indexes2sel=np.array(indexes2sel)
for line in lines:
 cols=line.split()
 if(cols):
  cols2=[float(c) for c in cols]
  cols2=np.array(cols2)
  cols2=cols2[indexes2sel]
  cols2=list(cols2)
#  print(cols2[7:])
#  print('lc2,vim:',len(cols2),vecInputNum+2)
#  print('cvin:',cols2[vecInputNum+1])
  if(len(cols2) == vecInputNum+2 and cols2[vecInputNum+1]>-1):
    idx_a=int(cols2[vecInputNum+1])
    if(not (idx_a in classes_sel)):
      classes_sel[idx_a]=count_classes
      count_classes+=1
    if(idx_a in ASYMMETRY):
      ASYMMETRY[idx_a]+=1
    else:
      ASYMMETRY[idx_a]=1
    if(i>=MAX_TRAIN_SIZE):
     break
    if(ASYMMETRY[idx_a]<100000000000 and i<MAX_TRAIN_SIZE):
     i+=1
#     z1=cols2[:vecInputNum+1]
     if(MODE=="withHeff"):
      z1=cols2[:vecInputNum+1]
     else:
      z1=cols2[:6]+cols2[7:vecInputNum+2]
#     print("-:",len(z1))
#     quit()

#    z1[vecInputNum-1]=cols2[vecInputNum+1]
#     z2=tf.keras.utils.to_categorical(int(cols2[vecInputNum]),num_classes=datesNum).tolist()
     XARR1.append(z1)
     DATES1.append(z1[vecInputNum:vecInputNum+1])
#     XARR1_INP.append(z1[4:7]+z1[8:vecInputNum])
     if(MODE=="withHeff"):
#      XARR1_INP.append(z1[4:vecInputNum])
      z1poly=pf.fit_transform([z1[4:vecInputNum]])[0] 
     else:
      z1poly=pf.fit_transform([z1[4:vecInputNum-1]])[0] 
#      XARR1_INP.append(z1[4:vecInputNum-1])
     XARR1_INP.append(z1poly)

#     print('z',z1[4:7])
     if(i%10000==0):
       print(int(cols2[vecInputNum]))
#    XARR1.append(tf.keras.utils.to_categorical(int(cols2[vecInputNum+1]),num_classes=output_class_M).tolist())
#     YARR1.append(tf.keras.utils.to_categorical(idx_a,num_classes=output_class_M).tolist())
     Cl_Y.append(idx_a)
     if(len(XARR1)>=MAX_BLOCK_SIZE):
      print("write block")
      XARR1_INP=np.array(XARR1_INP)
#      print(XARR1_INP)
      zz=model.predict(XARR1_INP)
#      print ("calc hidden classes for dataset:",zz)
#      quit()
      codes=tf.argmax(zz,axis=1)
      codes_fxd=np.array(codes)
#      print(codes)
#      print(codes_fxd)
#      quit()
      for i in range(len(codes)):
       id1=int(codes[i])
       zz_cur=zz[i]
       val_cur=zz_cur[id1]
       zz_cur[id1]=0
       next_code=tf.argmax(zz_cur)
       val_prev=zz_cur[next_code]
# correct not sure
#       if(val_cur<DETECTION_ACCURACY):
#        codes_fxd[i]=totalClassNum+id1


#        id1=int(codes_fxd[i])
#       if(id1==6):
#        if(abs(XARR1_INP[i][0])>100):
#         id1=17
#        else:
#         if(abs(XARR1_INP[i][2])>120):
#          id1=18
#       if(id1==4):
#        if(abs(XARR1_INP[i][0])>50):
#         id1=19
#       if(id1==11):
#        if((abs(XARR1_INP[i][0])>350 and abs(XARR1_INP[i][0])<550) or
#           (abs(XARR1_INP[i][0])>800 and abs(XARR1_INP[i][0])<870) or
#           (abs(XARR1_INP[i][0])>1350 and abs(XARR1_INP[i][0])<1430)
#           ):
#         id1=20
#       codes_fxd[i]=id1

       if id1 in stat:
        stat[id1]+=1
       else:
        stat[id1]=1
#      codes=codes_fxd.copy()
      codes=tf.convert_to_tensor(codes_fxd)
#     print ("hidden classes statistics:")
#     print(stat)

#     print ("prepare to save new dataset...")
      import numpy as np
      xarr = np.array(XARR1)[:,:vecInputNum]
      dates = np.array(DATES1)[:,:1]


      codes2=np.array([[c] for c in codes])

#      print(xarr.shape)
#      print(dates.shape)
#      print(codes2.shape)
#      quit()

#print(yp2[:5])

#     print ("concatenate values together...")

      newarray = np.concatenate((xarr,dates,codes2),axis=1)
# np.set_printoptions(precision=4)
#     print(newarray[:10])
      np.savetxt(fout,newarray, fmt='%.2f', delimiter='\t')
      XARR1=[]
      XARR1_INP=[]
      YARR1=[]
      DATES1=[]

print('classes:',classes_sel)
num=i-1
# print(XARR1.shape())
# random.shuffle(XARR1)
# random.shuffle(YARR1)

del lines

# print(XARRtrain)

#XARRval=np.array(random.sample(XARR1,int(num*0.1)))
#YARRval=np.array(YARR1[int(num*0.5):])


#YARRval=np.array(random.sample(YARR1,int(num*0.1)))

f.close()
fout.close()




print ("save ne dataset...")

# !gzip processed-Full-p_le_e-4-vars_15-sm.dat

