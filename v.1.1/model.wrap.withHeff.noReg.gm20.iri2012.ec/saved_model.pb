�
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
�
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.7.02v2.7.0-rc1-69-gc256c071bb28Ƿ
�
ExperimentEmb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameExperimentEmb/embeddings
�
,ExperimentEmb/embeddings/Read/ReadVariableOpReadVariableOpExperimentEmb/embeddings* 
_output_shapes
:
��*
dtype0
�
actualOutputClasses/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameactualOutputClasses/kernel
�
.actualOutputClasses/kernel/Read/ReadVariableOpReadVariableOpactualOutputClasses/kernel*
_output_shapes

:*
dtype0
�
actualOutputClasses/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameactualOutputClasses/bias
�
,actualOutputClasses/bias/Read/ReadVariableOpReadVariableOpactualOutputClasses/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

transform0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	B�*"
shared_nametransform0/kernel
x
%transform0/kernel/Read/ReadVariableOpReadVariableOptransform0/kernel*
_output_shapes
:	B�*
dtype0
w
transform0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nametransform0/bias
p
#transform0/bias/Read/ReadVariableOpReadVariableOptransform0/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:�*
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:�*
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:�*
dtype0
�
transform1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*"
shared_nametransform1/kernel
y
%transform1/kernel/Read/ReadVariableOpReadVariableOptransform1/kernel* 
_output_shapes
:
��*
dtype0
w
transform1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_nametransform1/bias
p
#transform1/bias/Read/ReadVariableOpReadVariableOptransform1/bias*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:�*
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:�*
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:�*
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:�*
dtype0
�
secondaryClasses/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*(
shared_namesecondaryClasses/kernel
�
+secondaryClasses/kernel/Read/ReadVariableOpReadVariableOpsecondaryClasses/kernel*
_output_shapes
:	�*
dtype0
�
secondaryClasses/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namesecondaryClasses/bias
{
)secondaryClasses/bias/Read/ReadVariableOpReadVariableOpsecondaryClasses/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
y
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nametrue_positives_1
r
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes	
:�*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:�*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:�*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:�*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:�*
dtype0
�
Adam/ExperimentEmb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!Adam/ExperimentEmb/embeddings/m
�
3Adam/ExperimentEmb/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/ExperimentEmb/embeddings/m* 
_output_shapes
:
��*
dtype0
�
!Adam/actualOutputClasses/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/actualOutputClasses/kernel/m
�
5Adam/actualOutputClasses/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/actualOutputClasses/kernel/m*
_output_shapes

:*
dtype0
�
Adam/actualOutputClasses/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/actualOutputClasses/bias/m
�
3Adam/actualOutputClasses/bias/m/Read/ReadVariableOpReadVariableOpAdam/actualOutputClasses/bias/m*
_output_shapes
:*
dtype0
�
Adam/transform0/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	B�*)
shared_nameAdam/transform0/kernel/m
�
,Adam/transform0/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transform0/kernel/m*
_output_shapes
:	B�*
dtype0
�
Adam/transform0/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/transform0/bias/m
~
*Adam/transform0/bias/m/Read/ReadVariableOpReadVariableOpAdam/transform0/bias/m*
_output_shapes	
:�*
dtype0
�
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/batch_normalization/gamma/m
�
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:�*
dtype0
�
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/batch_normalization/beta/m
�
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/transform1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/transform1/kernel/m
�
,Adam/transform1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transform1/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/transform1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/transform1/bias/m
~
*Adam/transform1/bias/m/Read/ReadVariableOpReadVariableOpAdam/transform1/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_1/gamma/m
�
6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_1/beta/m
�
5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/secondaryClasses/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/secondaryClasses/kernel/m
�
2Adam/secondaryClasses/kernel/m/Read/ReadVariableOpReadVariableOpAdam/secondaryClasses/kernel/m*
_output_shapes
:	�*
dtype0
�
Adam/secondaryClasses/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/secondaryClasses/bias/m
�
0Adam/secondaryClasses/bias/m/Read/ReadVariableOpReadVariableOpAdam/secondaryClasses/bias/m*
_output_shapes
:*
dtype0
�
Adam/ExperimentEmb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!Adam/ExperimentEmb/embeddings/v
�
3Adam/ExperimentEmb/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/ExperimentEmb/embeddings/v* 
_output_shapes
:
��*
dtype0
�
!Adam/actualOutputClasses/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/actualOutputClasses/kernel/v
�
5Adam/actualOutputClasses/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/actualOutputClasses/kernel/v*
_output_shapes

:*
dtype0
�
Adam/actualOutputClasses/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/actualOutputClasses/bias/v
�
3Adam/actualOutputClasses/bias/v/Read/ReadVariableOpReadVariableOpAdam/actualOutputClasses/bias/v*
_output_shapes
:*
dtype0
�
Adam/transform0/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	B�*)
shared_nameAdam/transform0/kernel/v
�
,Adam/transform0/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transform0/kernel/v*
_output_shapes
:	B�*
dtype0
�
Adam/transform0/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/transform0/bias/v
~
*Adam/transform0/bias/v/Read/ReadVariableOpReadVariableOpAdam/transform0/bias/v*
_output_shapes	
:�*
dtype0
�
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/batch_normalization/gamma/v
�
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:�*
dtype0
�
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/batch_normalization/beta/v
�
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/transform1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_nameAdam/transform1/kernel/v
�
,Adam/transform1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transform1/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/transform1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_nameAdam/transform1/bias/v
~
*Adam/transform1/bias/v/Read/ReadVariableOpReadVariableOpAdam/transform1/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*3
shared_name$"Adam/batch_normalization_1/gamma/v
�
6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:�*
dtype0
�
!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/batch_normalization_1/beta/v
�
5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/secondaryClasses/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*/
shared_name Adam/secondaryClasses/kernel/v
�
2Adam/secondaryClasses/kernel/v/Read/ReadVariableOpReadVariableOpAdam/secondaryClasses/kernel/v*
_output_shapes
:	�*
dtype0
�
Adam/secondaryClasses/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/secondaryClasses/bias/v
�
0Adam/secondaryClasses/bias/v/Read/ReadVariableOpReadVariableOpAdam/secondaryClasses/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�\
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�[
value�[B�[ B�[
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
 

	keras_api
b

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
 layer_with_weights-2
 layer-3
!layer_with_weights-3
!layer-4
"layer_with_weights-4
"layer-5
#	variables
$trainable_variables
%regularization_losses
&	keras_api
R
'	variables
(trainable_variables
)regularization_losses
*	keras_api
h

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
�
1iter

2beta_1

3beta_2
	4decay
5learning_ratem�+m�,m�6m�7m�8m�9m�<m�=m�>m�?m�Bm�Cm�v�+v�,v�6v�7v�8v�9v�<v�=v�>v�?v�Bv�Cv�
~
0
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
+15
,16
^
0
61
72
83
94
<5
=6
>7
?8
B9
C10
+11
,12
 
�
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics

	variables
trainable_variables
regularization_losses
 
 
hf
VARIABLE_VALUEExperimentEmb/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
 
h

6kernel
7bias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
�
\axis
	8gamma
9beta
:moving_mean
;moving_variance
]	variables
^trainable_variables
_regularization_losses
`	keras_api
h

<kernel
=bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
�
eaxis
	>gamma
?beta
@moving_mean
Amoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

Bkernel
Cbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
f
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
C13
F
60
71
82
93
<4
=5
>6
?7
B8
C9
 
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
#	variables
$trainable_variables
%regularization_losses
 
 
 
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
'	variables
(trainable_variables
)regularization_losses
fd
VARIABLE_VALUEactualOutputClasses/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEactualOutputClasses/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1

+0
,1
 
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
-	variables
.trainable_variables
/regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEtransform0/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtransform0/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/4/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/5/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/6/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEtransform1/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtransform1/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/gamma&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/beta'variables/10/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/11/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsecondaryClasses/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEsecondaryClasses/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
@2
A3
8
0
1
2
3
4
5
6
7

}0
~1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

60
71

60
71
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
 

80
91
:2
;3

80
91
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses

<0
=1

<0
=1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
 

>0
?1
@2
A3

>0
?1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses

B0
C1

B0
C1
 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses

:0
;1
@2
A3
*
0
1
2
 3
!4
"5
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
v
�true_positives
�true_negatives
�false_positives
�false_negatives
�	variables
�	keras_api
v
�true_positives
�true_negatives
�false_positives
�false_negatives
�	variables
�	keras_api
 
 
 
 
 

:0
;1
 
 
 
 
 
 
 
 
 

@0
A1
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
�0
�1
�2
�3

�	variables
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
�0
�1
�2
�3

�	variables
��
VARIABLE_VALUEAdam/ExperimentEmb/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/actualOutputClasses/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/actualOutputClasses/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform0/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform0/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/batch_normalization/gamma/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/batch_normalization/beta/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform1/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform1/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/secondaryClasses/kernel/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/secondaryClasses/bias/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/ExperimentEmb/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE!Adam/actualOutputClasses/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/actualOutputClasses/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform0/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform0/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/batch_normalization/gamma/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/batch_normalization/beta/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform1/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform1/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/secondaryClasses/kernel/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/secondaryClasses/bias/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_vec_datePlaceholder*'
_output_shapes
:���������G*
dtype0*
shape:���������G
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_vec_dateExperimentEmb/embeddingstransform0/kerneltransform0/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betatransform1/kerneltransform1/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betasecondaryClasses/kernelsecondaryClasses/biasactualOutputClasses/kernelactualOutputClasses/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2669714
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,ExperimentEmb/embeddings/Read/ReadVariableOp.actualOutputClasses/kernel/Read/ReadVariableOp,actualOutputClasses/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp%transform0/kernel/Read/ReadVariableOp#transform0/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp%transform1/kernel/Read/ReadVariableOp#transform1/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp+secondaryClasses/kernel/Read/ReadVariableOp)secondaryClasses/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp3Adam/ExperimentEmb/embeddings/m/Read/ReadVariableOp5Adam/actualOutputClasses/kernel/m/Read/ReadVariableOp3Adam/actualOutputClasses/bias/m/Read/ReadVariableOp,Adam/transform0/kernel/m/Read/ReadVariableOp*Adam/transform0/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp,Adam/transform1/kernel/m/Read/ReadVariableOp*Adam/transform1/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp2Adam/secondaryClasses/kernel/m/Read/ReadVariableOp0Adam/secondaryClasses/bias/m/Read/ReadVariableOp3Adam/ExperimentEmb/embeddings/v/Read/ReadVariableOp5Adam/actualOutputClasses/kernel/v/Read/ReadVariableOp3Adam/actualOutputClasses/bias/v/Read/ReadVariableOp,Adam/transform0/kernel/v/Read/ReadVariableOp*Adam/transform0/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp,Adam/transform1/kernel/v/Read/ReadVariableOp*Adam/transform1/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp2Adam/secondaryClasses/kernel/v/Read/ReadVariableOp0Adam/secondaryClasses/bias/v/Read/ReadVariableOpConst*G
Tin@
>2<	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2670712
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameExperimentEmb/embeddingsactualOutputClasses/kernelactualOutputClasses/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetransform0/kerneltransform0/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancetransform1/kerneltransform1/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancesecondaryClasses/kernelsecondaryClasses/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1true_negatives_1false_positives_1false_negatives_1Adam/ExperimentEmb/embeddings/m!Adam/actualOutputClasses/kernel/mAdam/actualOutputClasses/bias/mAdam/transform0/kernel/mAdam/transform0/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/transform1/kernel/mAdam/transform1/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/secondaryClasses/kernel/mAdam/secondaryClasses/bias/mAdam/ExperimentEmb/embeddings/v!Adam/actualOutputClasses/kernel/vAdam/actualOutputClasses/bias/vAdam/transform0/kernel/vAdam/transform0/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/transform1/kernel/vAdam/transform1/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/secondaryClasses/kernel/vAdam/secondaryClasses/bias/v*F
Tin?
=2;*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2670896��
�

�
G__inference_transform1_layer_call_and_return_conditional_losses_2668906

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������R
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������g
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�z
�
D__inference_model_1_layer_call_and_return_conditional_losses_2669885

inputs:
&experimentemb_embedding_lookup_2669804:
��B
/model_transform0_matmul_readvariableop_resource:	B�?
0model_transform0_biasadd_readvariableop_resource:	�J
;model_batch_normalization_batchnorm_readvariableop_resource:	�N
?model_batch_normalization_batchnorm_mul_readvariableop_resource:	�L
=model_batch_normalization_batchnorm_readvariableop_1_resource:	�L
=model_batch_normalization_batchnorm_readvariableop_2_resource:	�C
/model_transform1_matmul_readvariableop_resource:
��?
0model_transform1_biasadd_readvariableop_resource:	�L
=model_batch_normalization_1_batchnorm_readvariableop_resource:	�P
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�N
?model_batch_normalization_1_batchnorm_readvariableop_1_resource:	�N
?model_batch_normalization_1_batchnorm_readvariableop_2_resource:	�H
5model_secondaryclasses_matmul_readvariableop_resource:	�D
6model_secondaryclasses_biasadd_readvariableop_resource:D
2actualoutputclasses_matmul_readvariableop_resource:A
3actualoutputclasses_biasadd_readvariableop_resource:
identity��ExperimentEmb/embedding_lookup�*actualOutputClasses/BiasAdd/ReadVariableOp�)actualOutputClasses/MatMul/ReadVariableOp�2model/batch_normalization/batchnorm/ReadVariableOp�4model/batch_normalization/batchnorm/ReadVariableOp_1�4model/batch_normalization/batchnorm/ReadVariableOp_2�6model/batch_normalization/batchnorm/mul/ReadVariableOp�4model/batch_normalization_1/batchnorm/ReadVariableOp�6model/batch_normalization_1/batchnorm/ReadVariableOp_1�6model/batch_normalization_1/batchnorm/ReadVariableOp_2�8model/batch_normalization_1/batchnorm/mul/ReadVariableOp�-model/secondaryClasses/BiasAdd/ReadVariableOp�,model/secondaryClasses/MatMul/ReadVariableOp�'model/transform0/BiasAdd/ReadVariableOp�&model/transform0/MatMul/ReadVariableOp�'model/transform1/BiasAdd/ReadVariableOp�&model/transform1/MatMul/ReadVariableOpo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.split/splitSplitVinputstf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*�
_output_shapest
r:���������:���������:���������:���������:���������B:���������*
	num_splitt
ExperimentEmb/CastCasttf.split/split:output:5*

DstT0*

SrcT0*'
_output_shapes
:����������
ExperimentEmb/embedding_lookupResourceGather&experimentemb_embedding_lookup_2669804ExperimentEmb/Cast:y:0*
Tindices0*9
_class/
-+loc:@ExperimentEmb/embedding_lookup/2669804*,
_output_shapes
:����������*
dtype0�
'ExperimentEmb/embedding_lookup/IdentityIdentity'ExperimentEmb/embedding_lookup:output:0*
T0*9
_class/
-+loc:@ExperimentEmb/embedding_lookup/2669804*,
_output_shapes
:�����������
)ExperimentEmb/embedding_lookup/Identity_1Identity0ExperimentEmb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������o
reshape/ShapeShape2ExperimentEmb/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshape2ExperimentEmb/embedding_lookup/Identity_1:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:���������r
getOnlyPositiveValues/ReluRelureshape/Reshape:output:0*
T0*+
_output_shapes
:����������
&model/transform0/MatMul/ReadVariableOpReadVariableOp/model_transform0_matmul_readvariableop_resource*
_output_shapes
:	B�*
dtype0�
model/transform0/MatMulMatMultf.split/split:output:4.model/transform0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model/transform0/BiasAdd/ReadVariableOpReadVariableOp0model_transform0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/transform0/BiasAddBiasAdd!model/transform0/MatMul:product:0/model/transform0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
model/transform0/LeakyRelu	LeakyRelu!model/transform0/BiasAdd:output:0*(
_output_shapes
:�����������
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/mul_1Mul(model/transform0/LeakyRelu:activations:0+model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
&model/transform1/MatMul/ReadVariableOpReadVariableOp/model_transform1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/transform1/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0.model/transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model/transform1/BiasAdd/ReadVariableOpReadVariableOp0model_transform1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/transform1/BiasAddBiasAdd!model/transform1/MatMul:product:0/model/transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
model/transform1/LeakyRelu	LeakyRelu!model/transform1/BiasAdd:output:0*(
_output_shapes
:�����������
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/mul_1Mul(model/transform1/LeakyRelu:activations:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,model/secondaryClasses/MatMul/ReadVariableOpReadVariableOp5model_secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/secondaryClasses/MatMulMatMul/model/batch_normalization_1/batchnorm/add_1:z:04model/secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model/secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp6model_secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/secondaryClasses/BiasAddBiasAdd'model/secondaryClasses/MatMul:product:05model/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model/secondaryClasses/SoftmaxSoftmax'model/secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:���������T
dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims
ExpandDims(model/secondaryClasses/Softmax:softmax:0dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:����������

dot/MatMulBatchMatMulV2(getOnlyPositiveValues/Relu:activations:0dot/ExpandDims:output:0*
T0*+
_output_shapes
:���������L
	dot/ShapeShapedot/MatMul:output:0*
T0*
_output_shapes
:}
dot/SqueezeSqueezedot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims

����������
)actualOutputClasses/MatMul/ReadVariableOpReadVariableOp2actualoutputclasses_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
actualOutputClasses/MatMulMatMuldot/Squeeze:output:01actualOutputClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*actualOutputClasses/BiasAdd/ReadVariableOpReadVariableOp3actualoutputclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
actualOutputClasses/BiasAddBiasAdd$actualOutputClasses/MatMul:product:02actualOutputClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
actualOutputClasses/SoftmaxSoftmax$actualOutputClasses/BiasAdd:output:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%actualOutputClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^ExperimentEmb/embedding_lookup+^actualOutputClasses/BiasAdd/ReadVariableOp*^actualOutputClasses/MatMul/ReadVariableOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp.^model/secondaryClasses/BiasAdd/ReadVariableOp-^model/secondaryClasses/MatMul/ReadVariableOp(^model/transform0/BiasAdd/ReadVariableOp'^model/transform0/MatMul/ReadVariableOp(^model/transform1/BiasAdd/ReadVariableOp'^model/transform1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 2@
ExperimentEmb/embedding_lookupExperimentEmb/embedding_lookup2X
*actualOutputClasses/BiasAdd/ReadVariableOp*actualOutputClasses/BiasAdd/ReadVariableOp2V
)actualOutputClasses/MatMul/ReadVariableOp)actualOutputClasses/MatMul/ReadVariableOp2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2l
4model/batch_normalization/batchnorm/ReadVariableOp_14model/batch_normalization/batchnorm/ReadVariableOp_12l
4model/batch_normalization/batchnorm/ReadVariableOp_24model/batch_normalization/batchnorm/ReadVariableOp_22p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2p
6model/batch_normalization_1/batchnorm/ReadVariableOp_16model/batch_normalization_1/batchnorm/ReadVariableOp_12p
6model/batch_normalization_1/batchnorm/ReadVariableOp_26model/batch_normalization_1/batchnorm/ReadVariableOp_22t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2^
-model/secondaryClasses/BiasAdd/ReadVariableOp-model/secondaryClasses/BiasAdd/ReadVariableOp2\
,model/secondaryClasses/MatMul/ReadVariableOp,model/secondaryClasses/MatMul/ReadVariableOp2R
'model/transform0/BiasAdd/ReadVariableOp'model/transform0/BiasAdd/ReadVariableOp2P
&model/transform0/MatMul/ReadVariableOp&model/transform0/MatMul/ReadVariableOp2R
'model/transform1/BiasAdd/ReadVariableOp'model/transform1/BiasAdd/ReadVariableOp2P
&model/transform1/MatMul/ReadVariableOp&model/transform1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������G
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_2670341

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2668769p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�#
�
D__inference_model_1_layer_call_and_return_conditional_losses_2669667
vec_date)
experimentemb_2669626:
�� 
model_2669631:	B�
model_2669633:	�
model_2669635:	�
model_2669637:	�
model_2669639:	�
model_2669641:	�!
model_2669643:
��
model_2669645:	�
model_2669647:	�
model_2669649:	�
model_2669651:	�
model_2669653:	� 
model_2669655:	�
model_2669657:-
actualoutputclasses_2669661:)
actualoutputclasses_2669663:
identity��%ExperimentEmb/StatefulPartitionedCall�+actualOutputClasses/StatefulPartitionedCall�model/StatefulPartitionedCallo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.split/splitSplitVvec_datetf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*�
_output_shapest
r:���������:���������:���������:���������:���������B:���������*
	num_split�
%ExperimentEmb/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:5experimentemb_2669626*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2669235�
reshape/PartitionedCallPartitionedCall.ExperimentEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2669252�
%getOnlyPositiveValues/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2669259�
model/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:4model_2669631model_2669633model_2669635model_2669637model_2669639model_2669641model_2669643model_2669645model_2669647model_2669649model_2669651model_2669653model_2669655model_2669657*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2669072�
dot/PartitionedCallPartitionedCall.getOnlyPositiveValues/PartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2669300�
+actualOutputClasses/StatefulPartitionedCallStatefulPartitionedCalldot/PartitionedCall:output:0actualoutputclasses_2669661actualoutputclasses_2669663*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2669313�
IdentityIdentity4actualOutputClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^ExperimentEmb/StatefulPartitionedCall,^actualOutputClasses/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 2N
%ExperimentEmb/StatefulPartitionedCall%ExperimentEmb/StatefulPartitionedCall2Z
+actualOutputClasses/StatefulPartitionedCall+actualOutputClasses/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:Q M
'
_output_shapes
:���������G
"
_user_specified_name
vec_date
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2670461

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
2__inference_secondaryClasses_layer_call_fn_2670504

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2668932o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_2669563
vec_date
unknown:
��
	unknown_0:	B�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
��
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:

unknown_14:

unknown_15:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvec_dateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2669487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������G
"
_user_specified_name
vec_date
�N
�
B__inference_model_layer_call_and_return_conditional_losses_2670174

inputs<
)transform0_matmul_readvariableop_resource:	B�9
*transform0_biasadd_readvariableop_resource:	�D
5batch_normalization_batchnorm_readvariableop_resource:	�H
9batch_normalization_batchnorm_mul_readvariableop_resource:	�F
7batch_normalization_batchnorm_readvariableop_1_resource:	�F
7batch_normalization_batchnorm_readvariableop_2_resource:	�=
)transform1_matmul_readvariableop_resource:
��9
*transform1_biasadd_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	�H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	�B
/secondaryclasses_matmul_readvariableop_resource:	�>
0secondaryclasses_biasadd_readvariableop_resource:
identity��,batch_normalization/batchnorm/ReadVariableOp�.batch_normalization/batchnorm/ReadVariableOp_1�.batch_normalization/batchnorm/ReadVariableOp_2�0batch_normalization/batchnorm/mul/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�0batch_normalization_1/batchnorm/ReadVariableOp_1�0batch_normalization_1/batchnorm/ReadVariableOp_2�2batch_normalization_1/batchnorm/mul/ReadVariableOp�'secondaryClasses/BiasAdd/ReadVariableOp�&secondaryClasses/MatMul/ReadVariableOp�!transform0/BiasAdd/ReadVariableOp� transform0/MatMul/ReadVariableOp�!transform1/BiasAdd/ReadVariableOp� transform1/MatMul/ReadVariableOp�
 transform0/MatMul/ReadVariableOpReadVariableOp)transform0_matmul_readvariableop_resource*
_output_shapes
:	B�*
dtype0�
transform0/MatMulMatMulinputs(transform0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!transform0/BiasAdd/ReadVariableOpReadVariableOp*transform0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
transform0/BiasAddBiasAddtransform0/MatMul:product:0)transform0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������h
transform0/LeakyRelu	LeakyRelutransform0/BiasAdd:output:0*(
_output_shapes
:�����������
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_1Mul"transform0/LeakyRelu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
 transform1/MatMul/ReadVariableOpReadVariableOp)transform1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
transform1/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0(transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!transform1/BiasAdd/ReadVariableOpReadVariableOp*transform1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
transform1/BiasAddBiasAddtransform1/MatMul:product:0)transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������h
transform1/LeakyRelu	LeakyRelutransform1/BiasAdd:output:0*(
_output_shapes
:�����������
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_1/batchnorm/mul_1Mul"transform1/LeakyRelu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
&secondaryClasses/MatMul/ReadVariableOpReadVariableOp/secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
secondaryClasses/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0.secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp0secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
secondaryClasses/BiasAddBiasAdd!secondaryClasses/MatMul:product:0/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
secondaryClasses/SoftmaxSoftmax!secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"secondaryClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp(^secondaryClasses/BiasAdd/ReadVariableOp'^secondaryClasses/MatMul/ReadVariableOp"^transform0/BiasAdd/ReadVariableOp!^transform0/MatMul/ReadVariableOp"^transform1/BiasAdd/ReadVariableOp!^transform1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2d
0batch_normalization_1/batchnorm/ReadVariableOp_10batch_normalization_1/batchnorm/ReadVariableOp_12d
0batch_normalization_1/batchnorm/ReadVariableOp_20batch_normalization_1/batchnorm/ReadVariableOp_22h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2R
'secondaryClasses/BiasAdd/ReadVariableOp'secondaryClasses/BiasAdd/ReadVariableOp2P
&secondaryClasses/MatMul/ReadVariableOp&secondaryClasses/MatMul/ReadVariableOp2F
!transform0/BiasAdd/ReadVariableOp!transform0/BiasAdd/ReadVariableOp2D
 transform0/MatMul/ReadVariableOp transform0/MatMul/ReadVariableOp2F
!transform1/BiasAdd/ReadVariableOp!transform1/BiasAdd/ReadVariableOp2D
 transform1/MatMul/ReadVariableOp transform1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_2669357
vec_date
unknown:
��
	unknown_0:	B�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
��
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:

unknown_14:

unknown_15:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvec_dateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2669320o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������G
"
_user_specified_name
vec_date
�#
�
D__inference_model_1_layer_call_and_return_conditional_losses_2669487

inputs)
experimentemb_2669446:
�� 
model_2669451:	B�
model_2669453:	�
model_2669455:	�
model_2669457:	�
model_2669459:	�
model_2669461:	�!
model_2669463:
��
model_2669465:	�
model_2669467:	�
model_2669469:	�
model_2669471:	�
model_2669473:	� 
model_2669475:	�
model_2669477:-
actualoutputclasses_2669481:)
actualoutputclasses_2669483:
identity��%ExperimentEmb/StatefulPartitionedCall�+actualOutputClasses/StatefulPartitionedCall�model/StatefulPartitionedCallo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.split/splitSplitVinputstf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*�
_output_shapest
r:���������:���������:���������:���������:���������B:���������*
	num_split�
%ExperimentEmb/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:5experimentemb_2669446*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2669235�
reshape/PartitionedCallPartitionedCall.ExperimentEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2669252�
%getOnlyPositiveValues/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2669259�
model/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:4model_2669451model_2669453model_2669455model_2669457model_2669459model_2669461model_2669463model_2669465model_2669467model_2669469model_2669471model_2669473model_2669475model_2669477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2669072�
dot/PartitionedCallPartitionedCall.getOnlyPositiveValues/PartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2669300�
+actualOutputClasses/StatefulPartitionedCallStatefulPartitionedCalldot/PartitionedCall:output:0actualoutputclasses_2669481actualoutputclasses_2669483*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2669313�
IdentityIdentity4actualOutputClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^ExperimentEmb/StatefulPartitionedCall,^actualOutputClasses/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 2N
%ExperimentEmb/StatefulPartitionedCall%ExperimentEmb/StatefulPartitionedCall2Z
+actualOutputClasses/StatefulPartitionedCall+actualOutputClasses/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:O K
'
_output_shapes
:���������G
 
_user_specified_nameinputs
�
S
7__inference_getOnlyPositiveValues_layer_call_fn_2670046

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2669259d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_model_layer_call_and_return_conditional_losses_2670259

inputs<
)transform0_matmul_readvariableop_resource:	B�9
*transform0_biasadd_readvariableop_resource:	�J
;batch_normalization_assignmovingavg_readvariableop_resource:	�L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	�H
9batch_normalization_batchnorm_mul_readvariableop_resource:	�D
5batch_normalization_batchnorm_readvariableop_resource:	�=
)transform1_matmul_readvariableop_resource:
��9
*transform1_biasadd_readvariableop_resource:	�L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	�N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	�F
7batch_normalization_1_batchnorm_readvariableop_resource:	�B
/secondaryclasses_matmul_readvariableop_resource:	�>
0secondaryclasses_biasadd_readvariableop_resource:
identity��#batch_normalization/AssignMovingAvg�2batch_normalization/AssignMovingAvg/ReadVariableOp�%batch_normalization/AssignMovingAvg_1�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�,batch_normalization/batchnorm/ReadVariableOp�0batch_normalization/batchnorm/mul/ReadVariableOp�%batch_normalization_1/AssignMovingAvg�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�'batch_normalization_1/AssignMovingAvg_1�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�.batch_normalization_1/batchnorm/ReadVariableOp�2batch_normalization_1/batchnorm/mul/ReadVariableOp�'secondaryClasses/BiasAdd/ReadVariableOp�&secondaryClasses/MatMul/ReadVariableOp�!transform0/BiasAdd/ReadVariableOp� transform0/MatMul/ReadVariableOp�!transform1/BiasAdd/ReadVariableOp� transform1/MatMul/ReadVariableOp�
 transform0/MatMul/ReadVariableOpReadVariableOp)transform0_matmul_readvariableop_resource*
_output_shapes
:	B�*
dtype0�
transform0/MatMulMatMulinputs(transform0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!transform0/BiasAdd/ReadVariableOpReadVariableOp*transform0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
transform0/BiasAddBiasAddtransform0/MatMul:product:0)transform0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������h
transform0/LeakyRelu	LeakyRelutransform0/BiasAdd:output:0*(
_output_shapes
:����������|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
 batch_normalization/moments/meanMean"transform0/LeakyRelu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
-batch_normalization/moments/SquaredDifferenceSquaredDifference"transform0/LeakyRelu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
#batch_normalization/AssignMovingAvgAssignSubVariableOp;batch_normalization_assignmovingavg_readvariableop_resource+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0p
+batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/mul_1Mul"transform0/LeakyRelu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
 transform1/MatMul/ReadVariableOpReadVariableOp)transform1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
transform1/MatMulMatMul'batch_normalization/batchnorm/add_1:z:0(transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
!transform1/BiasAdd/ReadVariableOpReadVariableOp*transform1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
transform1/BiasAddBiasAddtransform1/MatMul:product:0)transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������h
transform1/LeakyRelu	LeakyRelutransform1/BiasAdd:output:0*(
_output_shapes
:����������~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"batch_normalization_1/moments/meanMean"transform1/LeakyRelu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	��
/batch_normalization_1/moments/SquaredDifferenceSquaredDifference"transform1/LeakyRelu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
%batch_normalization_1/AssignMovingAvgAssignSubVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0r
-batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
%batch_normalization_1/batchnorm/mul_1Mul"transform1/LeakyRelu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
&secondaryClasses/MatMul/ReadVariableOpReadVariableOp/secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
secondaryClasses/MatMulMatMul)batch_normalization_1/batchnorm/add_1:z:0.secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp0secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
secondaryClasses/BiasAddBiasAdd!secondaryClasses/MatMul:product:0/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
secondaryClasses/SoftmaxSoftmax!secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"secondaryClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp(^secondaryClasses/BiasAdd/ReadVariableOp'^secondaryClasses/MatMul/ReadVariableOp"^transform0/BiasAdd/ReadVariableOp!^transform0/MatMul/ReadVariableOp"^transform1/BiasAdd/ReadVariableOp!^transform1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 2J
#batch_normalization/AssignMovingAvg#batch_normalization/AssignMovingAvg2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2N
%batch_normalization/AssignMovingAvg_1%batch_normalization/AssignMovingAvg_12l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2N
%batch_normalization_1/AssignMovingAvg%batch_normalization_1/AssignMovingAvg2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2R
'batch_normalization_1/AssignMovingAvg_1'batch_normalization_1/AssignMovingAvg_12p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2`
.batch_normalization_1/batchnorm/ReadVariableOp.batch_normalization_1/batchnorm/ReadVariableOp2h
2batch_normalization_1/batchnorm/mul/ReadVariableOp2batch_normalization_1/batchnorm/mul/ReadVariableOp2R
'secondaryClasses/BiasAdd/ReadVariableOp'secondaryClasses/BiasAdd/ReadVariableOp2P
&secondaryClasses/MatMul/ReadVariableOp&secondaryClasses/MatMul/ReadVariableOp2F
!transform0/BiasAdd/ReadVariableOp!transform0/BiasAdd/ReadVariableOp2D
 transform0/MatMul/ReadVariableOp transform0/MatMul/ReadVariableOp2F
!transform1/BiasAdd/ReadVariableOp!transform1/BiasAdd/ReadVariableOp2D
 transform1/MatMul/ReadVariableOp transform1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�

`
D__inference_reshape_layer_call_and_return_conditional_losses_2669252

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2670361

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2670515

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2670295

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
,__inference_transform0_layer_call_fn_2670304

inputs
unknown:	B�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform0_layer_call_and_return_conditional_losses_2668880p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������B: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�#
�
D__inference_model_1_layer_call_and_return_conditional_losses_2669320

inputs)
experimentemb_2669236:
�� 
model_2669261:	B�
model_2669263:	�
model_2669265:	�
model_2669267:	�
model_2669269:	�
model_2669271:	�!
model_2669273:
��
model_2669275:	�
model_2669277:	�
model_2669279:	�
model_2669281:	�
model_2669283:	� 
model_2669285:	�
model_2669287:-
actualoutputclasses_2669314:)
actualoutputclasses_2669316:
identity��%ExperimentEmb/StatefulPartitionedCall�+actualOutputClasses/StatefulPartitionedCall�model/StatefulPartitionedCallo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.split/splitSplitVinputstf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*�
_output_shapest
r:���������:���������:���������:���������:���������B:���������*
	num_split�
%ExperimentEmb/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:5experimentemb_2669236*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2669235�
reshape/PartitionedCallPartitionedCall.ExperimentEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2669252�
%getOnlyPositiveValues/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2669259�
model/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:4model_2669261model_2669263model_2669265model_2669267model_2669269model_2669271model_2669273model_2669275model_2669277model_2669279model_2669281model_2669283model_2669285model_2669287*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2668939�
dot/PartitionedCallPartitionedCall.getOnlyPositiveValues/PartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2669300�
+actualOutputClasses/StatefulPartitionedCallStatefulPartitionedCalldot/PartitionedCall:output:0actualoutputclasses_2669314actualoutputclasses_2669316*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2669313�
IdentityIdentity4actualOutputClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^ExperimentEmb/StatefulPartitionedCall,^actualOutputClasses/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 2N
%ExperimentEmb/StatefulPartitionedCall%ExperimentEmb/StatefulPartitionedCall2Z
+actualOutputClasses/StatefulPartitionedCall+actualOutputClasses/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:O K
'
_output_shapes
:���������G
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_2669714
vec_date
unknown:
��
	unknown_0:	B�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
��
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:

unknown_14:

unknown_15:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvec_dateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2668698o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������G
"
_user_specified_name
vec_date
�%
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2668851

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2668722

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
n
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2670051

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:���������^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
G__inference_transform1_layer_call_and_return_conditional_losses_2670415

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������R
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������g
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2668769

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
B__inference_model_layer_call_and_return_conditional_losses_2669173
	vec_input%
transform0_2669139:	B�!
transform0_2669141:	�*
batch_normalization_2669144:	�*
batch_normalization_2669146:	�*
batch_normalization_2669148:	�*
batch_normalization_2669150:	�&
transform1_2669153:
��!
transform1_2669155:	�,
batch_normalization_1_2669158:	�,
batch_normalization_1_2669160:	�,
batch_normalization_1_2669162:	�,
batch_normalization_1_2669164:	�+
secondaryclasses_2669167:	�&
secondaryclasses_2669169:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�(secondaryClasses/StatefulPartitionedCall�"transform0/StatefulPartitionedCall�"transform1/StatefulPartitionedCall�
"transform0/StatefulPartitionedCallStatefulPartitionedCall	vec_inputtransform0_2669139transform0_2669141*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform0_layer_call_and_return_conditional_losses_2668880�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+transform0/StatefulPartitionedCall:output:0batch_normalization_2669144batch_normalization_2669146batch_normalization_2669148batch_normalization_2669150*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2668722�
"transform1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0transform1_2669153transform1_2669155*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2668906�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall+transform1/StatefulPartitionedCall:output:0batch_normalization_1_2669158batch_normalization_1_2669160batch_normalization_1_2669162batch_normalization_1_2669164*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2668804�
(secondaryClasses/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0secondaryclasses_2669167secondaryclasses_2669169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2668932�
IdentityIdentity1secondaryClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall)^secondaryClasses/StatefulPartitionedCall#^transform0/StatefulPartitionedCall#^transform1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2T
(secondaryClasses/StatefulPartitionedCall(secondaryClasses/StatefulPartitionedCall2H
"transform0/StatefulPartitionedCall"transform0/StatefulPartitionedCall2H
"transform1/StatefulPartitionedCall"transform1/StatefulPartitionedCall:R N
'
_output_shapes
:���������B
#
_user_specified_name	vec_input
�

�
G__inference_transform0_layer_call_and_return_conditional_losses_2670315

inputs1
matmul_readvariableop_resource:	B�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	B�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������R
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������g
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������B: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
n
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2669259

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:���������^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
B__inference_model_layer_call_and_return_conditional_losses_2668939

inputs%
transform0_2668881:	B�!
transform0_2668883:	�*
batch_normalization_2668886:	�*
batch_normalization_2668888:	�*
batch_normalization_2668890:	�*
batch_normalization_2668892:	�&
transform1_2668907:
��!
transform1_2668909:	�,
batch_normalization_1_2668912:	�,
batch_normalization_1_2668914:	�,
batch_normalization_1_2668916:	�,
batch_normalization_1_2668918:	�+
secondaryclasses_2668933:	�&
secondaryclasses_2668935:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�(secondaryClasses/StatefulPartitionedCall�"transform0/StatefulPartitionedCall�"transform1/StatefulPartitionedCall�
"transform0/StatefulPartitionedCallStatefulPartitionedCallinputstransform0_2668881transform0_2668883*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform0_layer_call_and_return_conditional_losses_2668880�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+transform0/StatefulPartitionedCall:output:0batch_normalization_2668886batch_normalization_2668888batch_normalization_2668890batch_normalization_2668892*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2668722�
"transform1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0transform1_2668907transform1_2668909*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2668906�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall+transform1/StatefulPartitionedCall:output:0batch_normalization_1_2668912batch_normalization_1_2668914batch_normalization_1_2668916batch_normalization_1_2668918*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2668804�
(secondaryClasses/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0secondaryclasses_2668933secondaryclasses_2668935*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2668932�
IdentityIdentity1secondaryClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall)^secondaryClasses/StatefulPartitionedCall#^transform0/StatefulPartitionedCall#^transform1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2T
(secondaryClasses/StatefulPartitionedCall(secondaryClasses/StatefulPartitionedCall2H
"transform0/StatefulPartitionedCall"transform0/StatefulPartitionedCall2H
"transform1/StatefulPartitionedCall"transform1/StatefulPartitionedCall:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_2669792

inputs
unknown:
��
	unknown_0:	B�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
��
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:

unknown_14:

unknown_15:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2669487o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������G
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_layer_call_fn_2670328

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2668722p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_2670084

inputs
unknown:	B�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2668939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�

�
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2669313

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

`
D__inference_reshape_layer_call_and_return_conditional_losses_2670041

inputs
identity;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Q
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�!
�
B__inference_model_layer_call_and_return_conditional_losses_2669210
	vec_input%
transform0_2669176:	B�!
transform0_2669178:	�*
batch_normalization_2669181:	�*
batch_normalization_2669183:	�*
batch_normalization_2669185:	�*
batch_normalization_2669187:	�&
transform1_2669190:
��!
transform1_2669192:	�,
batch_normalization_1_2669195:	�,
batch_normalization_1_2669197:	�,
batch_normalization_1_2669199:	�,
batch_normalization_1_2669201:	�+
secondaryclasses_2669204:	�&
secondaryclasses_2669206:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�(secondaryClasses/StatefulPartitionedCall�"transform0/StatefulPartitionedCall�"transform1/StatefulPartitionedCall�
"transform0/StatefulPartitionedCallStatefulPartitionedCall	vec_inputtransform0_2669176transform0_2669178*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform0_layer_call_and_return_conditional_losses_2668880�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+transform0/StatefulPartitionedCall:output:0batch_normalization_2669181batch_normalization_2669183batch_normalization_2669185batch_normalization_2669187*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2668769�
"transform1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0transform1_2669190transform1_2669192*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2668906�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall+transform1/StatefulPartitionedCall:output:0batch_normalization_1_2669195batch_normalization_1_2669197batch_normalization_1_2669199batch_normalization_1_2669201*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2668851�
(secondaryClasses/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0secondaryclasses_2669204secondaryclasses_2669206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2668932�
IdentityIdentity1secondaryClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall)^secondaryClasses/StatefulPartitionedCall#^transform0/StatefulPartitionedCall#^transform1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2T
(secondaryClasses/StatefulPartitionedCall(secondaryClasses/StatefulPartitionedCall2H
"transform0/StatefulPartitionedCall"transform0/StatefulPartitionedCall2H
"transform1/StatefulPartitionedCall"transform1/StatefulPartitionedCall:R N
'
_output_shapes
:���������B
#
_user_specified_name	vec_input
�

�
G__inference_transform0_layer_call_and_return_conditional_losses_2668880

inputs1
matmul_readvariableop_resource:	B�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	B�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������R
	LeakyRelu	LeakyReluBiasAdd:output:0*(
_output_shapes
:����������g
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������B: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
��
�%
#__inference__traced_restore_2670896
file_prefix=
)assignvariableop_experimentemb_embeddings:
��?
-assignvariableop_1_actualoutputclasses_kernel:9
+assignvariableop_2_actualoutputclasses_bias:&
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: 7
$assignvariableop_8_transform0_kernel:	B�1
"assignvariableop_9_transform0_bias:	�<
-assignvariableop_10_batch_normalization_gamma:	�;
,assignvariableop_11_batch_normalization_beta:	�B
3assignvariableop_12_batch_normalization_moving_mean:	�F
7assignvariableop_13_batch_normalization_moving_variance:	�9
%assignvariableop_14_transform1_kernel:
��2
#assignvariableop_15_transform1_bias:	�>
/assignvariableop_16_batch_normalization_1_gamma:	�=
.assignvariableop_17_batch_normalization_1_beta:	�D
5assignvariableop_18_batch_normalization_1_moving_mean:	�H
9assignvariableop_19_batch_normalization_1_moving_variance:	�>
+assignvariableop_20_secondaryclasses_kernel:	�7
)assignvariableop_21_secondaryclasses_bias:#
assignvariableop_22_total: #
assignvariableop_23_count: 1
"assignvariableop_24_true_positives:	�1
"assignvariableop_25_true_negatives:	�2
#assignvariableop_26_false_positives:	�2
#assignvariableop_27_false_negatives:	�3
$assignvariableop_28_true_positives_1:	�3
$assignvariableop_29_true_negatives_1:	�4
%assignvariableop_30_false_positives_1:	�4
%assignvariableop_31_false_negatives_1:	�G
3assignvariableop_32_adam_experimentemb_embeddings_m:
��G
5assignvariableop_33_adam_actualoutputclasses_kernel_m:A
3assignvariableop_34_adam_actualoutputclasses_bias_m:?
,assignvariableop_35_adam_transform0_kernel_m:	B�9
*assignvariableop_36_adam_transform0_bias_m:	�C
4assignvariableop_37_adam_batch_normalization_gamma_m:	�B
3assignvariableop_38_adam_batch_normalization_beta_m:	�@
,assignvariableop_39_adam_transform1_kernel_m:
��9
*assignvariableop_40_adam_transform1_bias_m:	�E
6assignvariableop_41_adam_batch_normalization_1_gamma_m:	�D
5assignvariableop_42_adam_batch_normalization_1_beta_m:	�E
2assignvariableop_43_adam_secondaryclasses_kernel_m:	�>
0assignvariableop_44_adam_secondaryclasses_bias_m:G
3assignvariableop_45_adam_experimentemb_embeddings_v:
��G
5assignvariableop_46_adam_actualoutputclasses_kernel_v:A
3assignvariableop_47_adam_actualoutputclasses_bias_v:?
,assignvariableop_48_adam_transform0_kernel_v:	B�9
*assignvariableop_49_adam_transform0_bias_v:	�C
4assignvariableop_50_adam_batch_normalization_gamma_v:	�B
3assignvariableop_51_adam_batch_normalization_beta_v:	�@
,assignvariableop_52_adam_transform1_kernel_v:
��9
*assignvariableop_53_adam_transform1_bias_v:	�E
6assignvariableop_54_adam_batch_normalization_1_gamma_v:	�D
5assignvariableop_55_adam_batch_normalization_1_beta_v:	�E
2assignvariableop_56_adam_secondaryclasses_kernel_v:	�>
0assignvariableop_57_adam_secondaryclasses_bias_v:
identity_59��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B�;B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*I
dtypes?
=2;	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp)assignvariableop_experimentemb_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp-assignvariableop_1_actualoutputclasses_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp+assignvariableop_2_actualoutputclasses_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp$assignvariableop_8_transform0_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp"assignvariableop_9_transform0_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp-assignvariableop_10_batch_normalization_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp,assignvariableop_11_batch_normalization_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp3assignvariableop_12_batch_normalization_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp7assignvariableop_13_batch_normalization_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp%assignvariableop_14_transform1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp#assignvariableop_15_transform1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_1_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_1_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_1_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_1_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp+assignvariableop_20_secondaryclasses_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp)assignvariableop_21_secondaryclasses_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_totalIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_countIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp"assignvariableop_24_true_positivesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp"assignvariableop_25_true_negativesIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp#assignvariableop_26_false_positivesIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp#assignvariableop_27_false_negativesIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp$assignvariableop_28_true_positives_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp$assignvariableop_29_true_negatives_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp%assignvariableop_30_false_positives_1Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp%assignvariableop_31_false_negatives_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_experimentemb_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp5assignvariableop_33_adam_actualoutputclasses_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_actualoutputclasses_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp,assignvariableop_35_adam_transform0_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp*assignvariableop_36_adam_transform0_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp4assignvariableop_37_adam_batch_normalization_gamma_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_batch_normalization_beta_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp,assignvariableop_39_adam_transform1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp*assignvariableop_40_adam_transform1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp6assignvariableop_41_adam_batch_normalization_1_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp5assignvariableop_42_adam_batch_normalization_1_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp2assignvariableop_43_adam_secondaryclasses_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp0assignvariableop_44_adam_secondaryclasses_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp3assignvariableop_45_adam_experimentemb_embeddings_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp5assignvariableop_46_adam_actualoutputclasses_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp3assignvariableop_47_adam_actualoutputclasses_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp,assignvariableop_48_adam_transform0_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_transform0_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp4assignvariableop_50_adam_batch_normalization_gamma_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp3assignvariableop_51_adam_batch_normalization_beta_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp,assignvariableop_52_adam_transform1_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_transform1_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp6assignvariableop_54_adam_batch_normalization_1_gamma_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_batch_normalization_1_beta_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp2assignvariableop_56_adam_secondaryclasses_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp0assignvariableop_57_adam_secondaryclasses_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_58Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_59IdentityIdentity_58:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_59Identity_59:output:0*�
_input_shapesx
v: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
7__inference_batch_normalization_1_layer_call_fn_2670441

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2668851p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_actualOutputClasses_layer_call_fn_2670284

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2669313o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
j
@__inference_dot_layer_call_and_return_conditional_losses_2669300

inputs
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :q

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������j
MatMulBatchMatMulV2inputsExpandDims:output:0*
T0*+
_output_shapes
:���������D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:u
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims

���������X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������:���������:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
l
@__inference_dot_layer_call_and_return_conditional_losses_2670275
inputs_0
inputs_1
identityP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :q

ExpandDims
ExpandDimsinputs_1ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������l
MatMulBatchMatMulV2inputs_0ExpandDims:output:0*
T0*+
_output_shapes
:���������D
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:u
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims

���������X
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������:���������:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
'__inference_model_layer_call_fn_2669136
	vec_input
unknown:	B�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	vec_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2669072o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������B
#
_user_specified_name	vec_input
�
Q
%__inference_dot_layer_call_fn_2670265
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2669300`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������:���������:U Q
+
_output_shapes
:���������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�r
�
 __inference__traced_save_2670712
file_prefix7
3savev2_experimentemb_embeddings_read_readvariableop9
5savev2_actualoutputclasses_kernel_read_readvariableop7
3savev2_actualoutputclasses_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop0
,savev2_transform0_kernel_read_readvariableop.
*savev2_transform0_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop0
,savev2_transform1_kernel_read_readvariableop.
*savev2_transform1_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop6
2savev2_secondaryclasses_kernel_read_readvariableop4
0savev2_secondaryclasses_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop/
+savev2_true_negatives_1_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop>
:savev2_adam_experimentemb_embeddings_m_read_readvariableop@
<savev2_adam_actualoutputclasses_kernel_m_read_readvariableop>
:savev2_adam_actualoutputclasses_bias_m_read_readvariableop7
3savev2_adam_transform0_kernel_m_read_readvariableop5
1savev2_adam_transform0_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop7
3savev2_adam_transform1_kernel_m_read_readvariableop5
1savev2_adam_transform1_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop=
9savev2_adam_secondaryclasses_kernel_m_read_readvariableop;
7savev2_adam_secondaryclasses_bias_m_read_readvariableop>
:savev2_adam_experimentemb_embeddings_v_read_readvariableop@
<savev2_adam_actualoutputclasses_kernel_v_read_readvariableop>
:savev2_adam_actualoutputclasses_bias_v_read_readvariableop7
3savev2_adam_transform0_kernel_v_read_readvariableop5
1savev2_adam_transform0_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop7
3savev2_adam_transform1_kernel_v_read_readvariableop5
1savev2_adam_transform1_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop=
9savev2_adam_secondaryclasses_kernel_v_read_readvariableop;
7savev2_adam_secondaryclasses_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B�;B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:;*
dtype0*�
value�B~;B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_experimentemb_embeddings_read_readvariableop5savev2_actualoutputclasses_kernel_read_readvariableop3savev2_actualoutputclasses_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop,savev2_transform0_kernel_read_readvariableop*savev2_transform0_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop,savev2_transform1_kernel_read_readvariableop*savev2_transform1_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop2savev2_secondaryclasses_kernel_read_readvariableop0savev2_secondaryclasses_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop+savev2_true_negatives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop:savev2_adam_experimentemb_embeddings_m_read_readvariableop<savev2_adam_actualoutputclasses_kernel_m_read_readvariableop:savev2_adam_actualoutputclasses_bias_m_read_readvariableop3savev2_adam_transform0_kernel_m_read_readvariableop1savev2_adam_transform0_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop3savev2_adam_transform1_kernel_m_read_readvariableop1savev2_adam_transform1_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop9savev2_adam_secondaryclasses_kernel_m_read_readvariableop7savev2_adam_secondaryclasses_bias_m_read_readvariableop:savev2_adam_experimentemb_embeddings_v_read_readvariableop<savev2_adam_actualoutputclasses_kernel_v_read_readvariableop:savev2_adam_actualoutputclasses_bias_v_read_readvariableop3savev2_adam_transform0_kernel_v_read_readvariableop1savev2_adam_transform0_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop3savev2_adam_transform1_kernel_v_read_readvariableop1savev2_adam_transform1_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop9savev2_adam_secondaryclasses_kernel_v_read_readvariableop7savev2_adam_secondaryclasses_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *I
dtypes?
=2;	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��::: : : : : :	B�:�:�:�:�:�:
��:�:�:�:�:�:	�:: : :�:�:�:�:�:�:�:�:
��:::	B�:�:�:�:
��:�:�:�:	�::
��:::	B�:�:�:�:
��:�:�:�:	�:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%	!

_output_shapes
:	B�:!


_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:! 

_output_shapes	
:�:&!"
 
_output_shapes
:
��:$" 

_output_shapes

:: #

_output_shapes
::%$!

_output_shapes
:	B�:!%

_output_shapes	
:�:!&

_output_shapes	
:�:!'

_output_shapes	
:�:&("
 
_output_shapes
:
��:!)

_output_shapes	
:�:!*

_output_shapes	
:�:!+

_output_shapes	
:�:%,!

_output_shapes
:	�: -

_output_shapes
::&."
 
_output_shapes
:
��:$/ 

_output_shapes

:: 0

_output_shapes
::%1!

_output_shapes
:	B�:!2

_output_shapes	
:�:!3

_output_shapes	
:�:!4

_output_shapes	
:�:&5"
 
_output_shapes
:
��:!6

_output_shapes	
:�:!7

_output_shapes	
:�:!8

_output_shapes	
:�:%9!

_output_shapes
:	�: :

_output_shapes
::;

_output_shapes
: 
�
�
/__inference_ExperimentEmb_layer_call_fn_2670013

inputs
unknown:
��
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2669235t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
Ӊ
�
"__inference__wrapped_model_2668698
vec_dateB
.model_1_experimentemb_embedding_lookup_2668617:
��J
7model_1_model_transform0_matmul_readvariableop_resource:	B�G
8model_1_model_transform0_biasadd_readvariableop_resource:	�R
Cmodel_1_model_batch_normalization_batchnorm_readvariableop_resource:	�V
Gmodel_1_model_batch_normalization_batchnorm_mul_readvariableop_resource:	�T
Emodel_1_model_batch_normalization_batchnorm_readvariableop_1_resource:	�T
Emodel_1_model_batch_normalization_batchnorm_readvariableop_2_resource:	�K
7model_1_model_transform1_matmul_readvariableop_resource:
��G
8model_1_model_transform1_biasadd_readvariableop_resource:	�T
Emodel_1_model_batch_normalization_1_batchnorm_readvariableop_resource:	�X
Imodel_1_model_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�V
Gmodel_1_model_batch_normalization_1_batchnorm_readvariableop_1_resource:	�V
Gmodel_1_model_batch_normalization_1_batchnorm_readvariableop_2_resource:	�P
=model_1_model_secondaryclasses_matmul_readvariableop_resource:	�L
>model_1_model_secondaryclasses_biasadd_readvariableop_resource:L
:model_1_actualoutputclasses_matmul_readvariableop_resource:I
;model_1_actualoutputclasses_biasadd_readvariableop_resource:
identity��&model_1/ExperimentEmb/embedding_lookup�2model_1/actualOutputClasses/BiasAdd/ReadVariableOp�1model_1/actualOutputClasses/MatMul/ReadVariableOp�:model_1/model/batch_normalization/batchnorm/ReadVariableOp�<model_1/model/batch_normalization/batchnorm/ReadVariableOp_1�<model_1/model/batch_normalization/batchnorm/ReadVariableOp_2�>model_1/model/batch_normalization/batchnorm/mul/ReadVariableOp�<model_1/model/batch_normalization_1/batchnorm/ReadVariableOp�>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1�>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2�@model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp�5model_1/model/secondaryClasses/BiasAdd/ReadVariableOp�4model_1/model/secondaryClasses/MatMul/ReadVariableOp�/model_1/model/transform0/BiasAdd/ReadVariableOp�.model_1/model/transform0/MatMul/ReadVariableOp�/model_1/model/transform1/BiasAdd/ReadVariableOp�.model_1/model/transform1/MatMul/ReadVariableOpw
model_1/tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      b
 model_1/tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/tf.split/splitSplitVvec_datemodel_1/tf.split/Const:output:0)model_1/tf.split/split/split_dim:output:0*
T0*

Tlen0*�
_output_shapest
r:���������:���������:���������:���������:���������B:���������*
	num_split�
model_1/ExperimentEmb/CastCastmodel_1/tf.split/split:output:5*

DstT0*

SrcT0*'
_output_shapes
:����������
&model_1/ExperimentEmb/embedding_lookupResourceGather.model_1_experimentemb_embedding_lookup_2668617model_1/ExperimentEmb/Cast:y:0*
Tindices0*A
_class7
53loc:@model_1/ExperimentEmb/embedding_lookup/2668617*,
_output_shapes
:����������*
dtype0�
/model_1/ExperimentEmb/embedding_lookup/IdentityIdentity/model_1/ExperimentEmb/embedding_lookup:output:0*
T0*A
_class7
53loc:@model_1/ExperimentEmb/embedding_lookup/2668617*,
_output_shapes
:�����������
1model_1/ExperimentEmb/embedding_lookup/Identity_1Identity8model_1/ExperimentEmb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������
model_1/reshape/ShapeShape:model_1/ExperimentEmb/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:m
#model_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%model_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%model_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model_1/reshape/strided_sliceStridedSlicemodel_1/reshape/Shape:output:0,model_1/reshape/strided_slice/stack:output:0.model_1/reshape/strided_slice/stack_1:output:0.model_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
model_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :a
model_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
model_1/reshape/Reshape/shapePack&model_1/reshape/strided_slice:output:0(model_1/reshape/Reshape/shape/1:output:0(model_1/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
model_1/reshape/ReshapeReshape:model_1/ExperimentEmb/embedding_lookup/Identity_1:output:0&model_1/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:����������
"model_1/getOnlyPositiveValues/ReluRelu model_1/reshape/Reshape:output:0*
T0*+
_output_shapes
:����������
.model_1/model/transform0/MatMul/ReadVariableOpReadVariableOp7model_1_model_transform0_matmul_readvariableop_resource*
_output_shapes
:	B�*
dtype0�
model_1/model/transform0/MatMulMatMulmodel_1/tf.split/split:output:46model_1/model/transform0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/model_1/model/transform0/BiasAdd/ReadVariableOpReadVariableOp8model_1_model_transform0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 model_1/model/transform0/BiasAddBiasAdd)model_1/model/transform0/MatMul:product:07model_1/model/transform0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model_1/model/transform0/LeakyRelu	LeakyRelu)model_1/model/transform0/BiasAdd:output:0*(
_output_shapes
:�����������
:model_1/model/batch_normalization/batchnorm/ReadVariableOpReadVariableOpCmodel_1_model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0v
1model_1/model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
/model_1/model/batch_normalization/batchnorm/addAddV2Bmodel_1/model/batch_normalization/batchnorm/ReadVariableOp:value:0:model_1/model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
1model_1/model/batch_normalization/batchnorm/RsqrtRsqrt3model_1/model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
>model_1/model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpGmodel_1_model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/model_1/model/batch_normalization/batchnorm/mulMul5model_1/model/batch_normalization/batchnorm/Rsqrt:y:0Fmodel_1/model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
1model_1/model/batch_normalization/batchnorm/mul_1Mul0model_1/model/transform0/LeakyRelu:activations:03model_1/model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
<model_1/model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpEmodel_1_model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
1model_1/model/batch_normalization/batchnorm/mul_2MulDmodel_1/model/batch_normalization/batchnorm/ReadVariableOp_1:value:03model_1/model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
<model_1/model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpEmodel_1_model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
/model_1/model/batch_normalization/batchnorm/subSubDmodel_1/model/batch_normalization/batchnorm/ReadVariableOp_2:value:05model_1/model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
1model_1/model/batch_normalization/batchnorm/add_1AddV25model_1/model/batch_normalization/batchnorm/mul_1:z:03model_1/model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
.model_1/model/transform1/MatMul/ReadVariableOpReadVariableOp7model_1_model_transform1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_1/model/transform1/MatMulMatMul5model_1/model/batch_normalization/batchnorm/add_1:z:06model_1/model/transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
/model_1/model/transform1/BiasAdd/ReadVariableOpReadVariableOp8model_1_model_transform1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
 model_1/model/transform1/BiasAddBiasAdd)model_1/model/transform1/MatMul:product:07model_1/model/transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model_1/model/transform1/LeakyRelu	LeakyRelu)model_1/model/transform1/BiasAdd:output:0*(
_output_shapes
:�����������
<model_1/model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpEmodel_1_model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0x
3model_1/model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
1model_1/model/batch_normalization_1/batchnorm/addAddV2Dmodel_1/model/batch_normalization_1/batchnorm/ReadVariableOp:value:0<model_1/model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
3model_1/model/batch_normalization_1/batchnorm/RsqrtRsqrt5model_1/model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
@model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpImodel_1_model_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1model_1/model/batch_normalization_1/batchnorm/mulMul7model_1/model/batch_normalization_1/batchnorm/Rsqrt:y:0Hmodel_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
3model_1/model/batch_normalization_1/batchnorm/mul_1Mul0model_1/model/transform1/LeakyRelu:activations:05model_1/model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_1_model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
3model_1/model/batch_normalization_1/batchnorm/mul_2MulFmodel_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:05model_1/model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_1_model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
1model_1/model/batch_normalization_1/batchnorm/subSubFmodel_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:07model_1/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
3model_1/model/batch_normalization_1/batchnorm/add_1AddV27model_1/model/batch_normalization_1/batchnorm/mul_1:z:05model_1/model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
4model_1/model/secondaryClasses/MatMul/ReadVariableOpReadVariableOp=model_1_model_secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
%model_1/model/secondaryClasses/MatMulMatMul7model_1/model/batch_normalization_1/batchnorm/add_1:z:0<model_1/model/secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
5model_1/model/secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp>model_1_model_secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
&model_1/model/secondaryClasses/BiasAddBiasAdd/model_1/model/secondaryClasses/MatMul:product:0=model_1/model/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&model_1/model/secondaryClasses/SoftmaxSoftmax/model_1/model/secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:���������\
model_1/dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
model_1/dot/ExpandDims
ExpandDims0model_1/model/secondaryClasses/Softmax:softmax:0#model_1/dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:����������
model_1/dot/MatMulBatchMatMulV20model_1/getOnlyPositiveValues/Relu:activations:0model_1/dot/ExpandDims:output:0*
T0*+
_output_shapes
:���������\
model_1/dot/ShapeShapemodel_1/dot/MatMul:output:0*
T0*
_output_shapes
:�
model_1/dot/SqueezeSqueezemodel_1/dot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims

����������
1model_1/actualOutputClasses/MatMul/ReadVariableOpReadVariableOp:model_1_actualoutputclasses_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
"model_1/actualOutputClasses/MatMulMatMulmodel_1/dot/Squeeze:output:09model_1/actualOutputClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
2model_1/actualOutputClasses/BiasAdd/ReadVariableOpReadVariableOp;model_1_actualoutputclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
#model_1/actualOutputClasses/BiasAddBiasAdd,model_1/actualOutputClasses/MatMul:product:0:model_1/actualOutputClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
#model_1/actualOutputClasses/SoftmaxSoftmax,model_1/actualOutputClasses/BiasAdd:output:0*
T0*'
_output_shapes
:���������|
IdentityIdentity-model_1/actualOutputClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^model_1/ExperimentEmb/embedding_lookup3^model_1/actualOutputClasses/BiasAdd/ReadVariableOp2^model_1/actualOutputClasses/MatMul/ReadVariableOp;^model_1/model/batch_normalization/batchnorm/ReadVariableOp=^model_1/model/batch_normalization/batchnorm/ReadVariableOp_1=^model_1/model/batch_normalization/batchnorm/ReadVariableOp_2?^model_1/model/batch_normalization/batchnorm/mul/ReadVariableOp=^model_1/model/batch_normalization_1/batchnorm/ReadVariableOp?^model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1?^model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2A^model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp6^model_1/model/secondaryClasses/BiasAdd/ReadVariableOp5^model_1/model/secondaryClasses/MatMul/ReadVariableOp0^model_1/model/transform0/BiasAdd/ReadVariableOp/^model_1/model/transform0/MatMul/ReadVariableOp0^model_1/model/transform1/BiasAdd/ReadVariableOp/^model_1/model/transform1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 2P
&model_1/ExperimentEmb/embedding_lookup&model_1/ExperimentEmb/embedding_lookup2h
2model_1/actualOutputClasses/BiasAdd/ReadVariableOp2model_1/actualOutputClasses/BiasAdd/ReadVariableOp2f
1model_1/actualOutputClasses/MatMul/ReadVariableOp1model_1/actualOutputClasses/MatMul/ReadVariableOp2x
:model_1/model/batch_normalization/batchnorm/ReadVariableOp:model_1/model/batch_normalization/batchnorm/ReadVariableOp2|
<model_1/model/batch_normalization/batchnorm/ReadVariableOp_1<model_1/model/batch_normalization/batchnorm/ReadVariableOp_12|
<model_1/model/batch_normalization/batchnorm/ReadVariableOp_2<model_1/model/batch_normalization/batchnorm/ReadVariableOp_22�
>model_1/model/batch_normalization/batchnorm/mul/ReadVariableOp>model_1/model/batch_normalization/batchnorm/mul/ReadVariableOp2|
<model_1/model/batch_normalization_1/batchnorm/ReadVariableOp<model_1/model/batch_normalization_1/batchnorm/ReadVariableOp2�
>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_12�
>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_22�
@model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp@model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp2n
5model_1/model/secondaryClasses/BiasAdd/ReadVariableOp5model_1/model/secondaryClasses/BiasAdd/ReadVariableOp2l
4model_1/model/secondaryClasses/MatMul/ReadVariableOp4model_1/model/secondaryClasses/MatMul/ReadVariableOp2b
/model_1/model/transform0/BiasAdd/ReadVariableOp/model_1/model/transform0/BiasAdd/ReadVariableOp2`
.model_1/model/transform0/MatMul/ReadVariableOp.model_1/model/transform0/MatMul/ReadVariableOp2b
/model_1/model/transform1/BiasAdd/ReadVariableOp/model_1/model/transform1/BiasAdd/ReadVariableOp2`
.model_1/model/transform1/MatMul/ReadVariableOp.model_1/model/transform1/MatMul/ReadVariableOp:Q M
'
_output_shapes
:���������G
"
_user_specified_name
vec_date
�#
�
D__inference_model_1_layer_call_and_return_conditional_losses_2669615
vec_date)
experimentemb_2669574:
�� 
model_2669579:	B�
model_2669581:	�
model_2669583:	�
model_2669585:	�
model_2669587:	�
model_2669589:	�!
model_2669591:
��
model_2669593:	�
model_2669595:	�
model_2669597:	�
model_2669599:	�
model_2669601:	� 
model_2669603:	�
model_2669605:-
actualoutputclasses_2669609:)
actualoutputclasses_2669611:
identity��%ExperimentEmb/StatefulPartitionedCall�+actualOutputClasses/StatefulPartitionedCall�model/StatefulPartitionedCallo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.split/splitSplitVvec_datetf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*�
_output_shapest
r:���������:���������:���������:���������:���������B:���������*
	num_split�
%ExperimentEmb/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:5experimentemb_2669574*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2669235�
reshape/PartitionedCallPartitionedCall.ExperimentEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2669252�
%getOnlyPositiveValues/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2669259�
model/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:4model_2669579model_2669581model_2669583model_2669585model_2669587model_2669589model_2669591model_2669593model_2669595model_2669597model_2669599model_2669601model_2669603model_2669605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2668939�
dot/PartitionedCallPartitionedCall.getOnlyPositiveValues/PartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2669300�
+actualOutputClasses/StatefulPartitionedCallStatefulPartitionedCalldot/PartitionedCall:output:0actualoutputclasses_2669609actualoutputclasses_2669611*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2669313�
IdentityIdentity4actualOutputClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp&^ExperimentEmb/StatefulPartitionedCall,^actualOutputClasses/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 2N
%ExperimentEmb/StatefulPartitionedCall%ExperimentEmb/StatefulPartitionedCall2Z
+actualOutputClasses/StatefulPartitionedCall+actualOutputClasses/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:Q M
'
_output_shapes
:���������G
"
_user_specified_name
vec_date
�
�
7__inference_batch_normalization_1_layer_call_fn_2670428

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2668804p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
)__inference_model_1_layer_call_fn_2669753

inputs
unknown:
��
	unknown_0:	B�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�
	unknown_6:
��
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:	�

unknown_13:

unknown_14:

unknown_15:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*3
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2669320o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������G
 
_user_specified_nameinputs
�%
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2670495

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
D__inference_model_1_layer_call_and_return_conditional_losses_2670006

inputs:
&experimentemb_embedding_lookup_2669897:
��B
/model_transform0_matmul_readvariableop_resource:	B�?
0model_transform0_biasadd_readvariableop_resource:	�P
Amodel_batch_normalization_assignmovingavg_readvariableop_resource:	�R
Cmodel_batch_normalization_assignmovingavg_1_readvariableop_resource:	�N
?model_batch_normalization_batchnorm_mul_readvariableop_resource:	�J
;model_batch_normalization_batchnorm_readvariableop_resource:	�C
/model_transform1_matmul_readvariableop_resource:
��?
0model_transform1_biasadd_readvariableop_resource:	�R
Cmodel_batch_normalization_1_assignmovingavg_readvariableop_resource:	�T
Emodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	�P
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	�L
=model_batch_normalization_1_batchnorm_readvariableop_resource:	�H
5model_secondaryclasses_matmul_readvariableop_resource:	�D
6model_secondaryclasses_biasadd_readvariableop_resource:D
2actualoutputclasses_matmul_readvariableop_resource:A
3actualoutputclasses_biasadd_readvariableop_resource:
identity��ExperimentEmb/embedding_lookup�*actualOutputClasses/BiasAdd/ReadVariableOp�)actualOutputClasses/MatMul/ReadVariableOp�)model/batch_normalization/AssignMovingAvg�8model/batch_normalization/AssignMovingAvg/ReadVariableOp�+model/batch_normalization/AssignMovingAvg_1�:model/batch_normalization/AssignMovingAvg_1/ReadVariableOp�2model/batch_normalization/batchnorm/ReadVariableOp�6model/batch_normalization/batchnorm/mul/ReadVariableOp�+model/batch_normalization_1/AssignMovingAvg�:model/batch_normalization_1/AssignMovingAvg/ReadVariableOp�-model/batch_normalization_1/AssignMovingAvg_1�<model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�4model/batch_normalization_1/batchnorm/ReadVariableOp�8model/batch_normalization_1/batchnorm/mul/ReadVariableOp�-model/secondaryClasses/BiasAdd/ReadVariableOp�,model/secondaryClasses/MatMul/ReadVariableOp�'model/transform0/BiasAdd/ReadVariableOp�&model/transform0/MatMul/ReadVariableOp�'model/transform1/BiasAdd/ReadVariableOp�&model/transform1/MatMul/ReadVariableOpo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :�
tf.split/splitSplitVinputstf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*�
_output_shapest
r:���������:���������:���������:���������:���������B:���������*
	num_splitt
ExperimentEmb/CastCasttf.split/split:output:5*

DstT0*

SrcT0*'
_output_shapes
:����������
ExperimentEmb/embedding_lookupResourceGather&experimentemb_embedding_lookup_2669897ExperimentEmb/Cast:y:0*
Tindices0*9
_class/
-+loc:@ExperimentEmb/embedding_lookup/2669897*,
_output_shapes
:����������*
dtype0�
'ExperimentEmb/embedding_lookup/IdentityIdentity'ExperimentEmb/embedding_lookup:output:0*
T0*9
_class/
-+loc:@ExperimentEmb/embedding_lookup/2669897*,
_output_shapes
:�����������
)ExperimentEmb/embedding_lookup/Identity_1Identity0ExperimentEmb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������o
reshape/ShapeShape2ExperimentEmb/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:e
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Y
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
reshape/ReshapeReshape2ExperimentEmb/embedding_lookup/Identity_1:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:���������r
getOnlyPositiveValues/ReluRelureshape/Reshape:output:0*
T0*+
_output_shapes
:����������
&model/transform0/MatMul/ReadVariableOpReadVariableOp/model_transform0_matmul_readvariableop_resource*
_output_shapes
:	B�*
dtype0�
model/transform0/MatMulMatMultf.split/split:output:4.model/transform0/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model/transform0/BiasAdd/ReadVariableOpReadVariableOp0model_transform0_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/transform0/BiasAddBiasAdd!model/transform0/MatMul:product:0/model/transform0/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
model/transform0/LeakyRelu	LeakyRelu!model/transform0/BiasAdd:output:0*(
_output_shapes
:�����������
8model/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
&model/batch_normalization/moments/meanMean(model/transform0/LeakyRelu:activations:0Amodel/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
.model/batch_normalization/moments/StopGradientStopGradient/model/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	��
3model/batch_normalization/moments/SquaredDifferenceSquaredDifference(model/transform0/LeakyRelu:activations:07model/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
<model/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
*model/batch_normalization/moments/varianceMean7model/batch_normalization/moments/SquaredDifference:z:0Emodel/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
)model/batch_normalization/moments/SqueezeSqueeze/model/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
+model/batch_normalization/moments/Squeeze_1Squeeze3model/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
/model/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
8model/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpAmodel_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
-model/batch_normalization/AssignMovingAvg/subSub@model/batch_normalization/AssignMovingAvg/ReadVariableOp:value:02model/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
-model/batch_normalization/AssignMovingAvg/mulMul1model/batch_normalization/AssignMovingAvg/sub:z:08model/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
)model/batch_normalization/AssignMovingAvgAssignSubVariableOpAmodel_batch_normalization_assignmovingavg_readvariableop_resource1model/batch_normalization/AssignMovingAvg/mul:z:09^model/batch_normalization/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0v
1model/batch_normalization/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
:model/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpCmodel_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/model/batch_normalization/AssignMovingAvg_1/subSubBmodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:04model/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
/model/batch_normalization/AssignMovingAvg_1/mulMul3model/batch_normalization/AssignMovingAvg_1/sub:z:0:model/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization/AssignMovingAvg_1AssignSubVariableOpCmodel_batch_normalization_assignmovingavg_1_readvariableop_resource3model/batch_normalization/AssignMovingAvg_1/mul:z:0;^model/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
'model/batch_normalization/batchnorm/addAddV24model/batch_normalization/moments/Squeeze_1:output:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:��
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/mul_1Mul(model/transform0/LeakyRelu:activations:0+model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
)model/batch_normalization/batchnorm/mul_2Mul2model/batch_normalization/moments/Squeeze:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
'model/batch_normalization/batchnorm/subSub:model/batch_normalization/batchnorm/ReadVariableOp:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
&model/transform1/MatMul/ReadVariableOpReadVariableOp/model_transform1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/transform1/MatMulMatMul-model/batch_normalization/batchnorm/add_1:z:0.model/transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
'model/transform1/BiasAdd/ReadVariableOpReadVariableOp0model_transform1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/transform1/BiasAddBiasAdd!model/transform1/MatMul:product:0/model/transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������t
model/transform1/LeakyRelu	LeakyRelu!model/transform1/BiasAdd:output:0*(
_output_shapes
:�����������
:model/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(model/batch_normalization_1/moments/meanMean(model/transform1/LeakyRelu:activations:0Cmodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
0model/batch_normalization_1/moments/StopGradientStopGradient1model/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	��
5model/batch_normalization_1/moments/SquaredDifferenceSquaredDifference(model/transform1/LeakyRelu:activations:09model/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
>model/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
,model/batch_normalization_1/moments/varianceMean9model/batch_normalization_1/moments/SquaredDifference:z:0Gmodel/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
+model/batch_normalization_1/moments/SqueezeSqueeze1model/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
-model/batch_normalization_1/moments/Squeeze_1Squeeze5model/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 v
1model/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
:model/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpCmodel_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
/model/batch_normalization_1/AssignMovingAvg/subSubBmodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:04model/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
/model/batch_normalization_1/AssignMovingAvg/mulMul3model/batch_normalization_1/AssignMovingAvg/sub:z:0:model/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/AssignMovingAvgAssignSubVariableOpCmodel_batch_normalization_1_assignmovingavg_readvariableop_resource3model/batch_normalization_1/AssignMovingAvg/mul:z:0;^model/batch_normalization_1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0x
3model/batch_normalization_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
<model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpEmodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
1model/batch_normalization_1/AssignMovingAvg_1/subSubDmodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:06model/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
1model/batch_normalization_1/AssignMovingAvg_1/mulMul5model/batch_normalization_1/AssignMovingAvg_1/sub:z:0<model/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
-model/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpEmodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource5model/batch_normalization_1/AssignMovingAvg_1/mul:z:0=^model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
)model/batch_normalization_1/batchnorm/addAddV26model/batch_normalization_1/moments/Squeeze_1:output:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/mul_1Mul(model/transform1/LeakyRelu:activations:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
+model/batch_normalization_1/batchnorm/mul_2Mul4model/batch_normalization_1/moments/Squeeze:output:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
)model/batch_normalization_1/batchnorm/subSub<model/batch_normalization_1/batchnorm/ReadVariableOp:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,model/secondaryClasses/MatMul/ReadVariableOpReadVariableOp5model_secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
model/secondaryClasses/MatMulMatMul/model/batch_normalization_1/batchnorm/add_1:z:04model/secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-model/secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp6model_secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
model/secondaryClasses/BiasAddBiasAdd'model/secondaryClasses/MatMul:product:05model/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
model/secondaryClasses/SoftmaxSoftmax'model/secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:���������T
dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
dot/ExpandDims
ExpandDims(model/secondaryClasses/Softmax:softmax:0dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:����������

dot/MatMulBatchMatMulV2(getOnlyPositiveValues/Relu:activations:0dot/ExpandDims:output:0*
T0*+
_output_shapes
:���������L
	dot/ShapeShapedot/MatMul:output:0*
T0*
_output_shapes
:}
dot/SqueezeSqueezedot/MatMul:output:0*
T0*'
_output_shapes
:���������*
squeeze_dims

����������
)actualOutputClasses/MatMul/ReadVariableOpReadVariableOp2actualoutputclasses_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
actualOutputClasses/MatMulMatMuldot/Squeeze:output:01actualOutputClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
*actualOutputClasses/BiasAdd/ReadVariableOpReadVariableOp3actualoutputclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
actualOutputClasses/BiasAddBiasAdd$actualOutputClasses/MatMul:product:02actualOutputClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������~
actualOutputClasses/SoftmaxSoftmax$actualOutputClasses/BiasAdd:output:0*
T0*'
_output_shapes
:���������t
IdentityIdentity%actualOutputClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^ExperimentEmb/embedding_lookup+^actualOutputClasses/BiasAdd/ReadVariableOp*^actualOutputClasses/MatMul/ReadVariableOp*^model/batch_normalization/AssignMovingAvg9^model/batch_normalization/AssignMovingAvg/ReadVariableOp,^model/batch_normalization/AssignMovingAvg_1;^model/batch_normalization/AssignMovingAvg_1/ReadVariableOp3^model/batch_normalization/batchnorm/ReadVariableOp7^model/batch_normalization/batchnorm/mul/ReadVariableOp,^model/batch_normalization_1/AssignMovingAvg;^model/batch_normalization_1/AssignMovingAvg/ReadVariableOp.^model/batch_normalization_1/AssignMovingAvg_1=^model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp9^model/batch_normalization_1/batchnorm/mul/ReadVariableOp.^model/secondaryClasses/BiasAdd/ReadVariableOp-^model/secondaryClasses/MatMul/ReadVariableOp(^model/transform0/BiasAdd/ReadVariableOp'^model/transform0/MatMul/ReadVariableOp(^model/transform1/BiasAdd/ReadVariableOp'^model/transform1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:���������G: : : : : : : : : : : : : : : : : 2@
ExperimentEmb/embedding_lookupExperimentEmb/embedding_lookup2X
*actualOutputClasses/BiasAdd/ReadVariableOp*actualOutputClasses/BiasAdd/ReadVariableOp2V
)actualOutputClasses/MatMul/ReadVariableOp)actualOutputClasses/MatMul/ReadVariableOp2V
)model/batch_normalization/AssignMovingAvg)model/batch_normalization/AssignMovingAvg2t
8model/batch_normalization/AssignMovingAvg/ReadVariableOp8model/batch_normalization/AssignMovingAvg/ReadVariableOp2Z
+model/batch_normalization/AssignMovingAvg_1+model/batch_normalization/AssignMovingAvg_12x
:model/batch_normalization/AssignMovingAvg_1/ReadVariableOp:model/batch_normalization/AssignMovingAvg_1/ReadVariableOp2h
2model/batch_normalization/batchnorm/ReadVariableOp2model/batch_normalization/batchnorm/ReadVariableOp2p
6model/batch_normalization/batchnorm/mul/ReadVariableOp6model/batch_normalization/batchnorm/mul/ReadVariableOp2Z
+model/batch_normalization_1/AssignMovingAvg+model/batch_normalization_1/AssignMovingAvg2x
:model/batch_normalization_1/AssignMovingAvg/ReadVariableOp:model/batch_normalization_1/AssignMovingAvg/ReadVariableOp2^
-model/batch_normalization_1/AssignMovingAvg_1-model/batch_normalization_1/AssignMovingAvg_12|
<model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp<model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2l
4model/batch_normalization_1/batchnorm/ReadVariableOp4model/batch_normalization_1/batchnorm/ReadVariableOp2t
8model/batch_normalization_1/batchnorm/mul/ReadVariableOp8model/batch_normalization_1/batchnorm/mul/ReadVariableOp2^
-model/secondaryClasses/BiasAdd/ReadVariableOp-model/secondaryClasses/BiasAdd/ReadVariableOp2\
,model/secondaryClasses/MatMul/ReadVariableOp,model/secondaryClasses/MatMul/ReadVariableOp2R
'model/transform0/BiasAdd/ReadVariableOp'model/transform0/BiasAdd/ReadVariableOp2P
&model/transform0/MatMul/ReadVariableOp&model/transform0/MatMul/ReadVariableOp2R
'model/transform1/BiasAdd/ReadVariableOp'model/transform1/BiasAdd/ReadVariableOp2P
&model/transform1/MatMul/ReadVariableOp&model/transform1/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������G
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_2668970
	vec_input
unknown:	B�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	vec_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2668939o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:���������B
#
_user_specified_name	vec_input
�	
�
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2669235

inputs,
embedding_lookup_2669229:
��
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_2669229Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2669229*,
_output_shapes
:����������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/2669229*,
_output_shapes
:�����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:����������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�!
�
B__inference_model_layer_call_and_return_conditional_losses_2669072

inputs%
transform0_2669038:	B�!
transform0_2669040:	�*
batch_normalization_2669043:	�*
batch_normalization_2669045:	�*
batch_normalization_2669047:	�*
batch_normalization_2669049:	�&
transform1_2669052:
��!
transform1_2669054:	�,
batch_normalization_1_2669057:	�,
batch_normalization_1_2669059:	�,
batch_normalization_1_2669061:	�,
batch_normalization_1_2669063:	�+
secondaryclasses_2669066:	�&
secondaryclasses_2669068:
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�(secondaryClasses/StatefulPartitionedCall�"transform0/StatefulPartitionedCall�"transform1/StatefulPartitionedCall�
"transform0/StatefulPartitionedCallStatefulPartitionedCallinputstransform0_2669038transform0_2669040*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform0_layer_call_and_return_conditional_losses_2668880�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+transform0/StatefulPartitionedCall:output:0batch_normalization_2669043batch_normalization_2669045batch_normalization_2669047batch_normalization_2669049*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2668769�
"transform1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0transform1_2669052transform1_2669054*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2668906�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall+transform1/StatefulPartitionedCall:output:0batch_normalization_1_2669057batch_normalization_1_2669059batch_normalization_1_2669061batch_normalization_1_2669063*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2668851�
(secondaryClasses/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0secondaryclasses_2669066secondaryclasses_2669068*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2668932�
IdentityIdentity1secondaryClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall)^secondaryClasses/StatefulPartitionedCall#^transform0/StatefulPartitionedCall#^transform1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2T
(secondaryClasses/StatefulPartitionedCall(secondaryClasses/StatefulPartitionedCall2H
"transform0/StatefulPartitionedCall"transform0/StatefulPartitionedCall2H
"transform1/StatefulPartitionedCall"transform1/StatefulPartitionedCall:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�
�
,__inference_transform1_layer_call_fn_2670404

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2668906p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2668804

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_model_layer_call_fn_2670117

inputs
unknown:	B�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:
��
	unknown_6:	�
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:	�

unknown_12:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*,
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2669072o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:���������B: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������B
 
_user_specified_nameinputs
�

�
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2668932

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2670023

inputs,
embedding_lookup_2670017:
��
identity��embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:����������
embedding_lookupResourceGatherembedding_lookup_2670017Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2670017*,
_output_shapes
:����������*
dtype0�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/2670017*,
_output_shapes
:�����������
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:����������x
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:����������Y
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
E
)__inference_reshape_layer_call_fn_2670028

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2669252d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������:T P
,
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2670395

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
=
vec_date1
serving_default_vec_date:0���������GG
actualOutputClasses0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

	variables
trainable_variables
regularization_losses
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
�

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
 layer_with_weights-2
 layer-3
!layer_with_weights-3
!layer-4
"layer_with_weights-4
"layer-5
#	variables
$trainable_variables
%regularization_losses
&	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

+kernel
,bias
-	variables
.trainable_variables
/regularization_losses
0	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
1iter

2beta_1

3beta_2
	4decay
5learning_ratem�+m�,m�6m�7m�8m�9m�<m�=m�>m�?m�Bm�Cm�v�+v�,v�6v�7v�8v�9v�<v�=v�>v�?v�Bv�Cv�"
	optimizer
�
0
61
72
83
94
:5
;6
<7
=8
>9
?10
@11
A12
B13
C14
+15
,16"
trackable_list_wrapper
~
0
61
72
83
94
<5
=6
>7
?8
B9
C10
+11
,12"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Dnon_trainable_variables

Elayers
Fmetrics
Glayer_regularization_losses
Hlayer_metrics

	variables
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
"
_generic_user_object
,:*
��2ExperimentEmb/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nnon_trainable_variables

Olayers
Pmetrics
Qlayer_regularization_losses
Rlayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
	variables
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_tf_keras_input_layer
�

6kernel
7bias
X	variables
Ytrainable_variables
Zregularization_losses
[	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
\axis
	8gamma
9beta
:moving_mean
;moving_variance
]	variables
^trainable_variables
_regularization_losses
`	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

<kernel
=bias
a	variables
btrainable_variables
cregularization_losses
d	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
eaxis
	>gamma
?beta
@moving_mean
Amoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Bkernel
Cbias
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
C13"
trackable_list_wrapper
f
60
71
82
93
<4
=5
>6
?7
B8
C9"
trackable_list_wrapper
 "
trackable_list_wrapper
�
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics
#	variables
$trainable_variables
%regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
snon_trainable_variables

tlayers
umetrics
vlayer_regularization_losses
wlayer_metrics
'	variables
(trainable_variables
)regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
,:*2actualOutputClasses/kernel
&:$2actualOutputClasses/bias
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
-	variables
.trainable_variables
/regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
$:"	B�2transform0/kernel
:�2transform0/bias
(:&�2batch_normalization/gamma
':%�2batch_normalization/beta
0:.� (2batch_normalization/moving_mean
4:2� (2#batch_normalization/moving_variance
%:#
��2transform1/kernel
:�2transform1/bias
*:(�2batch_normalization_1/gamma
):'�2batch_normalization_1/beta
2:0� (2!batch_normalization_1/moving_mean
6:4� (2%batch_normalization_1/moving_variance
*:(	�2secondaryClasses/kernel
#:!2secondaryClasses/bias
<
:0
;1
@2
A3"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
5
}0
~1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
X	variables
Ytrainable_variables
Zregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
80
91
:2
;3"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
<
:0
;1
@2
A3"
trackable_list_wrapper
J
0
1
2
 3
!4
"5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
�
�true_positives
�true_negatives
�false_positives
�false_negatives
�	variables
�	keras_api"
_tf_keras_metric
�
�true_positives
�true_negatives
�false_positives
�false_negatives
�	variables
�	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
1:/
��2Adam/ExperimentEmb/embeddings/m
1:/2!Adam/actualOutputClasses/kernel/m
+:)2Adam/actualOutputClasses/bias/m
):'	B�2Adam/transform0/kernel/m
#:!�2Adam/transform0/bias/m
-:+�2 Adam/batch_normalization/gamma/m
,:*�2Adam/batch_normalization/beta/m
*:(
��2Adam/transform1/kernel/m
#:!�2Adam/transform1/bias/m
/:-�2"Adam/batch_normalization_1/gamma/m
.:,�2!Adam/batch_normalization_1/beta/m
/:-	�2Adam/secondaryClasses/kernel/m
(:&2Adam/secondaryClasses/bias/m
1:/
��2Adam/ExperimentEmb/embeddings/v
1:/2!Adam/actualOutputClasses/kernel/v
+:)2Adam/actualOutputClasses/bias/v
):'	B�2Adam/transform0/kernel/v
#:!�2Adam/transform0/bias/v
-:+�2 Adam/batch_normalization/gamma/v
,:*�2Adam/batch_normalization/beta/v
*:(
��2Adam/transform1/kernel/v
#:!�2Adam/transform1/bias/v
/:-�2"Adam/batch_normalization_1/gamma/v
.:,�2!Adam/batch_normalization_1/beta/v
/:-	�2Adam/secondaryClasses/kernel/v
(:&2Adam/secondaryClasses/bias/v
�2�
)__inference_model_1_layer_call_fn_2669357
)__inference_model_1_layer_call_fn_2669753
)__inference_model_1_layer_call_fn_2669792
)__inference_model_1_layer_call_fn_2669563�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
D__inference_model_1_layer_call_and_return_conditional_losses_2669885
D__inference_model_1_layer_call_and_return_conditional_losses_2670006
D__inference_model_1_layer_call_and_return_conditional_losses_2669615
D__inference_model_1_layer_call_and_return_conditional_losses_2669667�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
"__inference__wrapped_model_2668698vec_date"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_ExperimentEmb_layer_call_fn_2670013�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2670023�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
)__inference_reshape_layer_call_fn_2670028�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_reshape_layer_call_and_return_conditional_losses_2670041�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_getOnlyPositiveValues_layer_call_fn_2670046�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2670051�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_model_layer_call_fn_2668970
'__inference_model_layer_call_fn_2670084
'__inference_model_layer_call_fn_2670117
'__inference_model_layer_call_fn_2669136�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_model_layer_call_and_return_conditional_losses_2670174
B__inference_model_layer_call_and_return_conditional_losses_2670259
B__inference_model_layer_call_and_return_conditional_losses_2669173
B__inference_model_layer_call_and_return_conditional_losses_2669210�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_dot_layer_call_fn_2670265�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_dot_layer_call_and_return_conditional_losses_2670275�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_actualOutputClasses_layer_call_fn_2670284�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2670295�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_2669714vec_date"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
,__inference_transform0_layer_call_fn_2670304�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_transform0_layer_call_and_return_conditional_losses_2670315�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_layer_call_fn_2670328
5__inference_batch_normalization_layer_call_fn_2670341�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2670361
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2670395�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
,__inference_transform1_layer_call_fn_2670404�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_transform1_layer_call_and_return_conditional_losses_2670415�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
7__inference_batch_normalization_1_layer_call_fn_2670428
7__inference_batch_normalization_1_layer_call_fn_2670441�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2670461
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2670495�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
2__inference_secondaryClasses_layer_call_fn_2670504�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2670515�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2670023`/�,
%�"
 �
inputs���������
� "*�'
 �
0����������
� �
/__inference_ExperimentEmb_layer_call_fn_2670013S/�,
%�"
 �
inputs���������
� "������������
"__inference__wrapped_model_2668698�67;8:9<=A>@?BC+,1�.
'�$
"�
vec_date���������G
� "I�F
D
actualOutputClasses-�*
actualOutputClasses����������
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2670295\+,/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
5__inference_actualOutputClasses_layer_call_fn_2670284O+,/�,
%�"
 �
inputs���������
� "�����������
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2670461dA>@?4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2670495d@A>?4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
7__inference_batch_normalization_1_layer_call_fn_2670428WA>@?4�1
*�'
!�
inputs����������
p 
� "������������
7__inference_batch_normalization_1_layer_call_fn_2670441W@A>?4�1
*�'
!�
inputs����������
p
� "������������
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2670361d;8:94�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2670395d:;894�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
5__inference_batch_normalization_layer_call_fn_2670328W;8:94�1
*�'
!�
inputs����������
p 
� "������������
5__inference_batch_normalization_layer_call_fn_2670341W:;894�1
*�'
!�
inputs����������
p
� "������������
@__inference_dot_layer_call_and_return_conditional_losses_2670275�^�[
T�Q
O�L
&�#
inputs/0���������
"�
inputs/1���������
� "%�"
�
0���������
� �
%__inference_dot_layer_call_fn_2670265z^�[
T�Q
O�L
&�#
inputs/0���������
"�
inputs/1���������
� "�����������
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2670051`3�0
)�&
$�!
inputs���������
� ")�&
�
0���������
� �
7__inference_getOnlyPositiveValues_layer_call_fn_2670046S3�0
)�&
$�!
inputs���������
� "�����������
D__inference_model_1_layer_call_and_return_conditional_losses_2669615u67;8:9<=A>@?BC+,9�6
/�,
"�
vec_date���������G
p 

 
� "%�"
�
0���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_2669667u67:;89<=@A>?BC+,9�6
/�,
"�
vec_date���������G
p

 
� "%�"
�
0���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_2669885s67;8:9<=A>@?BC+,7�4
-�*
 �
inputs���������G
p 

 
� "%�"
�
0���������
� �
D__inference_model_1_layer_call_and_return_conditional_losses_2670006s67:;89<=@A>?BC+,7�4
-�*
 �
inputs���������G
p

 
� "%�"
�
0���������
� �
)__inference_model_1_layer_call_fn_2669357h67;8:9<=A>@?BC+,9�6
/�,
"�
vec_date���������G
p 

 
� "�����������
)__inference_model_1_layer_call_fn_2669563h67:;89<=@A>?BC+,9�6
/�,
"�
vec_date���������G
p

 
� "�����������
)__inference_model_1_layer_call_fn_2669753f67;8:9<=A>@?BC+,7�4
-�*
 �
inputs���������G
p 

 
� "�����������
)__inference_model_1_layer_call_fn_2669792f67:;89<=@A>?BC+,7�4
-�*
 �
inputs���������G
p

 
� "�����������
B__inference_model_layer_call_and_return_conditional_losses_2669173s67;8:9<=A>@?BC:�7
0�-
#� 
	vec_input���������B
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_2669210s67:;89<=@A>?BC:�7
0�-
#� 
	vec_input���������B
p

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_2670174p67;8:9<=A>@?BC7�4
-�*
 �
inputs���������B
p 

 
� "%�"
�
0���������
� �
B__inference_model_layer_call_and_return_conditional_losses_2670259p67:;89<=@A>?BC7�4
-�*
 �
inputs���������B
p

 
� "%�"
�
0���������
� �
'__inference_model_layer_call_fn_2668970f67;8:9<=A>@?BC:�7
0�-
#� 
	vec_input���������B
p 

 
� "�����������
'__inference_model_layer_call_fn_2669136f67:;89<=@A>?BC:�7
0�-
#� 
	vec_input���������B
p

 
� "�����������
'__inference_model_layer_call_fn_2670084c67;8:9<=A>@?BC7�4
-�*
 �
inputs���������B
p 

 
� "�����������
'__inference_model_layer_call_fn_2670117c67:;89<=@A>?BC7�4
-�*
 �
inputs���������B
p

 
� "�����������
D__inference_reshape_layer_call_and_return_conditional_losses_2670041a4�1
*�'
%�"
inputs����������
� ")�&
�
0���������
� �
)__inference_reshape_layer_call_fn_2670028T4�1
*�'
%�"
inputs����������
� "�����������
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2670515]BC0�-
&�#
!�
inputs����������
� "%�"
�
0���������
� �
2__inference_secondaryClasses_layer_call_fn_2670504PBC0�-
&�#
!�
inputs����������
� "�����������
%__inference_signature_wrapper_2669714�67;8:9<=A>@?BC+,=�:
� 
3�0
.
vec_date"�
vec_date���������G"I�F
D
actualOutputClasses-�*
actualOutputClasses����������
G__inference_transform0_layer_call_and_return_conditional_losses_2670315]67/�,
%�"
 �
inputs���������B
� "&�#
�
0����������
� �
,__inference_transform0_layer_call_fn_2670304P67/�,
%�"
 �
inputs���������B
� "������������
G__inference_transform1_layer_call_and_return_conditional_losses_2670415^<=0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
,__inference_transform1_layer_call_fn_2670404Q<=0�-
&�#
!�
inputs����������
� "�����������