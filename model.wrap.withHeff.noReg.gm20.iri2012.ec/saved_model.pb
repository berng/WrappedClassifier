¡
è¹
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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
dtypetype
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
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
<
Selu
features"T
activations"T"
Ttype:
2
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

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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.02v2.7.0-rc1-69-gc256c071bb28¨¹

ExperimentEmb/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
²*)
shared_nameExperimentEmb/embeddings

,ExperimentEmb/embeddings/Read/ReadVariableOpReadVariableOpExperimentEmb/embeddings* 
_output_shapes
:
²*
dtype0

actualOutputClasses/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameactualOutputClasses/kernel

.actualOutputClasses/kernel/Read/ReadVariableOpReadVariableOpactualOutputClasses/kernel*
_output_shapes

:*
dtype0

actualOutputClasses/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameactualOutputClasses/bias

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
transform1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	B*"
shared_nametransform1/kernel
x
%transform1/kernel/Read/ReadVariableOpReadVariableOptransform1/kernel*
_output_shapes
:	B*
dtype0
w
transform1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nametransform1/bias
p
#transform1/bias/Read/ReadVariableOpReadVariableOptransform1/bias*
_output_shapes	
:*
dtype0

transform2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nametransform2/kernel
y
%transform2/kernel/Read/ReadVariableOpReadVariableOptransform2/kernel* 
_output_shapes
:
*
dtype0
w
transform2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nametransform2/bias
p
#transform2/bias/Read/ReadVariableOpReadVariableOptransform2/bias*
_output_shapes	
:*
dtype0

transform3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nametransform3/kernel
y
%transform3/kernel/Read/ReadVariableOpReadVariableOptransform3/kernel* 
_output_shapes
:
*
dtype0
w
transform3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nametransform3/bias
p
#transform3/bias/Read/ReadVariableOpReadVariableOptransform3/bias*
_output_shapes	
:*
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:*
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:*
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:*
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:*
dtype0

transform4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nametransform4/kernel
y
%transform4/kernel/Read/ReadVariableOpReadVariableOptransform4/kernel* 
_output_shapes
:
*
dtype0
w
transform4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nametransform4/bias
p
#transform4/bias/Read/ReadVariableOpReadVariableOptransform4/bias*
_output_shapes	
:*
dtype0

transform5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_nametransform5/kernel
y
%transform5/kernel/Read/ReadVariableOpReadVariableOptransform5/kernel* 
_output_shapes
:
*
dtype0
w
transform5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nametransform5/bias
p
#transform5/bias/Read/ReadVariableOpReadVariableOptransform5/bias*
_output_shapes	
:*
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes	
:*
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes	
:*
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes	
:*
dtype0

secondaryClasses/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_namesecondaryClasses/kernel

+secondaryClasses/kernel/Read/ReadVariableOpReadVariableOpsecondaryClasses/kernel*
_output_shapes
:	*
dtype0

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
shape:È*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:È*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:È*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:È*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:È* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:È*
dtype0
y
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*!
shared_nametrue_positives_1
r
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes	
:È*
dtype0
y
true_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*!
shared_nametrue_negatives_1
r
$true_negatives_1/Read/ReadVariableOpReadVariableOptrue_negatives_1*
_output_shapes	
:È*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:È*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:È*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:È*
dtype0

Adam/ExperimentEmb/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
²*0
shared_name!Adam/ExperimentEmb/embeddings/m

3Adam/ExperimentEmb/embeddings/m/Read/ReadVariableOpReadVariableOpAdam/ExperimentEmb/embeddings/m* 
_output_shapes
:
²*
dtype0

!Adam/actualOutputClasses/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/actualOutputClasses/kernel/m

5Adam/actualOutputClasses/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/actualOutputClasses/kernel/m*
_output_shapes

:*
dtype0

Adam/actualOutputClasses/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/actualOutputClasses/bias/m

3Adam/actualOutputClasses/bias/m/Read/ReadVariableOpReadVariableOpAdam/actualOutputClasses/bias/m*
_output_shapes
:*
dtype0

Adam/transform1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	B*)
shared_nameAdam/transform1/kernel/m

,Adam/transform1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transform1/kernel/m*
_output_shapes
:	B*
dtype0

Adam/transform1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform1/bias/m
~
*Adam/transform1/bias/m/Read/ReadVariableOpReadVariableOpAdam/transform1/bias/m*
_output_shapes	
:*
dtype0

Adam/transform2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/transform2/kernel/m

,Adam/transform2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transform2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/transform2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform2/bias/m
~
*Adam/transform2/bias/m/Read/ReadVariableOpReadVariableOpAdam/transform2/bias/m*
_output_shapes	
:*
dtype0

Adam/transform3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/transform3/kernel/m

,Adam/transform3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transform3/kernel/m* 
_output_shapes
:
*
dtype0

Adam/transform3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform3/bias/m
~
*Adam/transform3/bias/m/Read/ReadVariableOpReadVariableOpAdam/transform3/bias/m*
_output_shapes	
:*
dtype0

 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m

4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes	
:*
dtype0

Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m

3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes	
:*
dtype0

Adam/transform4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/transform4/kernel/m

,Adam/transform4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transform4/kernel/m* 
_output_shapes
:
*
dtype0

Adam/transform4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform4/bias/m
~
*Adam/transform4/bias/m/Read/ReadVariableOpReadVariableOpAdam/transform4/bias/m*
_output_shapes	
:*
dtype0

Adam/transform5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/transform5/kernel/m

,Adam/transform5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/transform5/kernel/m* 
_output_shapes
:
*
dtype0

Adam/transform5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform5/bias/m
~
*Adam/transform5/bias/m/Read/ReadVariableOpReadVariableOpAdam/transform5/bias/m*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/m

6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/m*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/m

5Adam/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/m*
_output_shapes	
:*
dtype0

Adam/secondaryClasses/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/secondaryClasses/kernel/m

2Adam/secondaryClasses/kernel/m/Read/ReadVariableOpReadVariableOpAdam/secondaryClasses/kernel/m*
_output_shapes
:	*
dtype0

Adam/secondaryClasses/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/secondaryClasses/bias/m

0Adam/secondaryClasses/bias/m/Read/ReadVariableOpReadVariableOpAdam/secondaryClasses/bias/m*
_output_shapes
:*
dtype0

Adam/ExperimentEmb/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
²*0
shared_name!Adam/ExperimentEmb/embeddings/v

3Adam/ExperimentEmb/embeddings/v/Read/ReadVariableOpReadVariableOpAdam/ExperimentEmb/embeddings/v* 
_output_shapes
:
²*
dtype0

!Adam/actualOutputClasses/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*2
shared_name#!Adam/actualOutputClasses/kernel/v

5Adam/actualOutputClasses/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/actualOutputClasses/kernel/v*
_output_shapes

:*
dtype0

Adam/actualOutputClasses/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/actualOutputClasses/bias/v

3Adam/actualOutputClasses/bias/v/Read/ReadVariableOpReadVariableOpAdam/actualOutputClasses/bias/v*
_output_shapes
:*
dtype0

Adam/transform1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	B*)
shared_nameAdam/transform1/kernel/v

,Adam/transform1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transform1/kernel/v*
_output_shapes
:	B*
dtype0

Adam/transform1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform1/bias/v
~
*Adam/transform1/bias/v/Read/ReadVariableOpReadVariableOpAdam/transform1/bias/v*
_output_shapes	
:*
dtype0

Adam/transform2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/transform2/kernel/v

,Adam/transform2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transform2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/transform2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform2/bias/v
~
*Adam/transform2/bias/v/Read/ReadVariableOpReadVariableOpAdam/transform2/bias/v*
_output_shapes	
:*
dtype0

Adam/transform3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/transform3/kernel/v

,Adam/transform3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transform3/kernel/v* 
_output_shapes
:
*
dtype0

Adam/transform3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform3/bias/v
~
*Adam/transform3/bias/v/Read/ReadVariableOpReadVariableOpAdam/transform3/bias/v*
_output_shapes	
:*
dtype0

 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v

4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes	
:*
dtype0

Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v

3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes	
:*
dtype0

Adam/transform4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/transform4/kernel/v

,Adam/transform4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transform4/kernel/v* 
_output_shapes
:
*
dtype0

Adam/transform4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform4/bias/v
~
*Adam/transform4/bias/v/Read/ReadVariableOpReadVariableOpAdam/transform4/bias/v*
_output_shapes	
:*
dtype0

Adam/transform5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/transform5/kernel/v

,Adam/transform5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/transform5/kernel/v* 
_output_shapes
:
*
dtype0

Adam/transform5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/transform5/bias/v
~
*Adam/transform5/bias/v/Read/ReadVariableOpReadVariableOpAdam/transform5/bias/v*
_output_shapes	
:*
dtype0

"Adam/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_1/gamma/v

6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_1/gamma/v*
_output_shapes	
:*
dtype0

!Adam/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization_1/beta/v

5Adam/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization_1/beta/v*
_output_shapes	
:*
dtype0

Adam/secondaryClasses/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/secondaryClasses/kernel/v

2Adam/secondaryClasses/kernel/v/Read/ReadVariableOpReadVariableOpAdam/secondaryClasses/kernel/v*
_output_shapes
:	*
dtype0

Adam/secondaryClasses/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameAdam/secondaryClasses/bias/v

0Adam/secondaryClasses/bias/v/Read/ReadVariableOpReadVariableOpAdam/secondaryClasses/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ïz
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ªz
value zBz Bz
§
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
²
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
 layer_with_weights-2
 layer-3
!layer_with_weights-3
!layer-4
"layer-5
#layer_with_weights-4
#layer-6
$layer_with_weights-5
$layer-7
%layer_with_weights-6
%layer-8
&layer-9
'layer_with_weights-7
'layer-10
(	variables
)trainable_variables
*regularization_losses
+	keras_api
R
,	variables
-trainable_variables
.regularization_losses
/	keras_api
h

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
¼
6iter

7beta_1

8beta_2
	9decay
:learning_ratemá0mâ1mã;mä<må=mæ>mç?mè@méAmêBmëEmìFmíGmîHmïImðJmñMmòNmóvô0võ1vö;v÷<vø=vù>vú?vû@vüAvýBvþEvÿFvGvHvIvJvMvNv
®
0
;1
<2
=3
>4
?5
@6
A7
B8
C9
D10
E11
F12
G13
H14
I15
J16
K17
L18
M19
N20
021
122

0
;1
<2
=3
>4
?5
@6
A7
B8
E9
F10
G11
H12
I13
J14
M15
N16
017
118
 
­
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
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
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
 
h

;kernel
<bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
h

=kernel
>bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
h

?kernel
@bias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api

oaxis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
R
t	variables
utrainable_variables
vregularization_losses
w	keras_api
h

Ekernel
Fbias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
h

Gkernel
Hbias
|	variables
}trainable_variables
~regularization_losses
	keras_api

	axis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
	variables
trainable_variables
regularization_losses
	keras_api
V
	variables
trainable_variables
regularization_losses
	keras_api
l

Mkernel
Nbias
	variables
trainable_variables
regularization_losses
	keras_api

;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
N19
v
;0
<1
=2
>3
?4
@5
A6
B7
E8
F9
G10
H11
I12
J13
M14
N15
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
(	variables
)trainable_variables
*regularization_losses
 
 
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
,	variables
-trainable_variables
.regularization_losses
fd
VARIABLE_VALUEactualOutputClasses/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEactualOutputClasses/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
 
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
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
VARIABLE_VALUEtransform1/kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtransform1/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEtransform2/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtransform2/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEtransform3/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEtransform3/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEbatch_normalization/gamma&variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEbatch_normalization/beta&variables/8/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/9/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE#batch_normalization/moving_variance'variables/10/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEtransform4/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEtransform4/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEtransform5/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEtransform5/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEbatch_normalization_1/gamma'variables/15/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEbatch_normalization_1/beta'variables/16/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization_1/moving_mean'variables/17/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization_1/moving_variance'variables/18/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsecondaryClasses/kernel'variables/19/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEsecondaryClasses/bias'variables/20/.ATTRIBUTES/VARIABLE_VALUE

C0
D1
K2
L3
8
0
1
2
3
4
5
6
7

0
1
2
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
;0
<1

;0
<1
 
²
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
c	variables
dtrainable_variables
eregularization_losses

=0
>1

=0
>1
 
²
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
g	variables
htrainable_variables
iregularization_losses

?0
@1

?0
@1
 
²
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
k	variables
ltrainable_variables
mregularization_losses
 

A0
B1
C2
D3

A0
B1
 
²
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
p	variables
qtrainable_variables
rregularization_losses
 
 
 
²
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
t	variables
utrainable_variables
vregularization_losses

E0
F1

E0
F1
 
²
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
x	variables
ytrainable_variables
zregularization_losses

G0
H1

G0
H1
 
²
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
|	variables
}trainable_variables
~regularization_losses
 

I0
J1
K2
L3

I0
J1
 
µ
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
µ
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
	variables
trainable_variables
regularization_losses

M0
N1

M0
N1
 
µ
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
	variables
trainable_variables
regularization_losses

C0
D1
K2
L3
N
0
1
2
 3
!4
"5
#6
$7
%8
&9
'10
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

Ñtotal

Òcount
Ó	variables
Ô	keras_api
v
Õtrue_positives
Ötrue_negatives
×false_positives
Øfalse_negatives
Ù	variables
Ú	keras_api
v
Ûtrue_positives
Ütrue_negatives
Ýfalse_positives
Þfalse_negatives
ß	variables
à	keras_api
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
C0
D1
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
 
 

K0
L1
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ñ0
Ò1

Ó	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
Õ0
Ö1
×2
Ø3

Ù	variables
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtrue_negatives_1=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
Û0
Ü1
Ý2
Þ3

ß	variables

VARIABLE_VALUEAdam/ExperimentEmb/embeddings/mVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/actualOutputClasses/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/actualOutputClasses/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform1/kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform1/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform2/kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform2/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform3/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform3/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/batch_normalization/gamma/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/batch_normalization/beta/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/transform4/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/transform4/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/transform5/kernel/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/transform5/bias/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_1/beta/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/secondaryClasses/kernel/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/secondaryClasses/bias/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/ExperimentEmb/embeddings/vVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE!Adam/actualOutputClasses/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUEAdam/actualOutputClasses/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform1/kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform1/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform2/kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform2/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/transform3/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/transform3/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/batch_normalization/gamma/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/batch_normalization/beta/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/transform4/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/transform4/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/transform5/kernel/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/transform5/bias/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/batch_normalization_1/gamma/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/batch_normalization_1/beta/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/secondaryClasses/kernel/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/secondaryClasses/bias/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_vec_datePlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿG

StatefulPartitionedCallStatefulPartitionedCallserving_default_vec_dateExperimentEmb/embeddingstransform1/kerneltransform1/biastransform2/kerneltransform2/biastransform3/kerneltransform3/bias#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betatransform4/kerneltransform4/biastransform5/kerneltransform5/bias%batch_normalization_1/moving_variancebatch_normalization_1/gamma!batch_normalization_1/moving_meanbatch_normalization_1/betasecondaryClasses/kernelsecondaryClasses/biasactualOutputClasses/kernelactualOutputClasses/bias*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_2027360
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
·
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,ExperimentEmb/embeddings/Read/ReadVariableOp.actualOutputClasses/kernel/Read/ReadVariableOp,actualOutputClasses/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp%transform1/kernel/Read/ReadVariableOp#transform1/bias/Read/ReadVariableOp%transform2/kernel/Read/ReadVariableOp#transform2/bias/Read/ReadVariableOp%transform3/kernel/Read/ReadVariableOp#transform3/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp%transform4/kernel/Read/ReadVariableOp#transform4/bias/Read/ReadVariableOp%transform5/kernel/Read/ReadVariableOp#transform5/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp+secondaryClasses/kernel/Read/ReadVariableOp)secondaryClasses/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp$true_negatives_1/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp3Adam/ExperimentEmb/embeddings/m/Read/ReadVariableOp5Adam/actualOutputClasses/kernel/m/Read/ReadVariableOp3Adam/actualOutputClasses/bias/m/Read/ReadVariableOp,Adam/transform1/kernel/m/Read/ReadVariableOp*Adam/transform1/bias/m/Read/ReadVariableOp,Adam/transform2/kernel/m/Read/ReadVariableOp*Adam/transform2/bias/m/Read/ReadVariableOp,Adam/transform3/kernel/m/Read/ReadVariableOp*Adam/transform3/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp,Adam/transform4/kernel/m/Read/ReadVariableOp*Adam/transform4/bias/m/Read/ReadVariableOp,Adam/transform5/kernel/m/Read/ReadVariableOp*Adam/transform5/bias/m/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/m/Read/ReadVariableOp5Adam/batch_normalization_1/beta/m/Read/ReadVariableOp2Adam/secondaryClasses/kernel/m/Read/ReadVariableOp0Adam/secondaryClasses/bias/m/Read/ReadVariableOp3Adam/ExperimentEmb/embeddings/v/Read/ReadVariableOp5Adam/actualOutputClasses/kernel/v/Read/ReadVariableOp3Adam/actualOutputClasses/bias/v/Read/ReadVariableOp,Adam/transform1/kernel/v/Read/ReadVariableOp*Adam/transform1/bias/v/Read/ReadVariableOp,Adam/transform2/kernel/v/Read/ReadVariableOp*Adam/transform2/bias/v/Read/ReadVariableOp,Adam/transform3/kernel/v/Read/ReadVariableOp*Adam/transform3/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp,Adam/transform4/kernel/v/Read/ReadVariableOp*Adam/transform4/bias/v/Read/ReadVariableOp,Adam/transform5/kernel/v/Read/ReadVariableOp*Adam/transform5/bias/v/Read/ReadVariableOp6Adam/batch_normalization_1/gamma/v/Read/ReadVariableOp5Adam/batch_normalization_1/beta/v/Read/ReadVariableOp2Adam/secondaryClasses/kernel/v/Read/ReadVariableOp0Adam/secondaryClasses/bias/v/Read/ReadVariableOpConst*Y
TinR
P2N	*
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_2028694
Â
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameExperimentEmb/embeddingsactualOutputClasses/kernelactualOutputClasses/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetransform1/kerneltransform1/biastransform2/kerneltransform2/biastransform3/kerneltransform3/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancetransform4/kerneltransform4/biastransform5/kerneltransform5/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variancesecondaryClasses/kernelsecondaryClasses/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativestrue_positives_1true_negatives_1false_positives_1false_negatives_1Adam/ExperimentEmb/embeddings/m!Adam/actualOutputClasses/kernel/mAdam/actualOutputClasses/bias/mAdam/transform1/kernel/mAdam/transform1/bias/mAdam/transform2/kernel/mAdam/transform2/bias/mAdam/transform3/kernel/mAdam/transform3/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/transform4/kernel/mAdam/transform4/bias/mAdam/transform5/kernel/mAdam/transform5/bias/m"Adam/batch_normalization_1/gamma/m!Adam/batch_normalization_1/beta/mAdam/secondaryClasses/kernel/mAdam/secondaryClasses/bias/mAdam/ExperimentEmb/embeddings/v!Adam/actualOutputClasses/kernel/vAdam/actualOutputClasses/bias/vAdam/transform1/kernel/vAdam/transform1/bias/vAdam/transform2/kernel/vAdam/transform2/bias/vAdam/transform3/kernel/vAdam/transform3/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/transform4/kernel/vAdam/transform4/bias/vAdam/transform5/kernel/vAdam/transform5/bias/v"Adam/batch_normalization_1/gamma/v!Adam/batch_normalization_1/beta/vAdam/secondaryClasses/kernel/vAdam/secondaryClasses/bias/v*X
TinQ
O2M*
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_2028932Äæ
&
Ñ
D__inference_model_1_layer_call_and_return_conditional_losses_2026870

inputs)
experimentemb_2026774:
² 
model_2026799:	B
model_2026801:	!
model_2026803:

model_2026805:	!
model_2026807:

model_2026809:	
model_2026811:	
model_2026813:	
model_2026815:	
model_2026817:	!
model_2026819:

model_2026821:	!
model_2026823:

model_2026825:	
model_2026827:	
model_2026829:	
model_2026831:	
model_2026833:	 
model_2026835:	
model_2026837:-
actualoutputclasses_2026864:)
actualoutputclasses_2026866:
identity¢%ExperimentEmb/StatefulPartitionedCall¢+actualOutputClasses/StatefulPartitionedCall¢model/StatefulPartitionedCallo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
tf.split/splitSplitVinputstf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*
_output_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿB:ÿÿÿÿÿÿÿÿÿ*
	num_split
%ExperimentEmb/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:5experimentemb_2026774*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2026773ã
reshape/PartitionedCallPartitionedCall.ExperimentEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2026790ñ
%getOnlyPositiveValues/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2026797ª
model/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:4model_2026799model_2026801model_2026803model_2026805model_2026807model_2026809model_2026811model_2026813model_2026815model_2026817model_2026819model_2026821model_2026823model_2026825model_2026827model_2026829model_2026831model_2026833model_2026835model_2026837* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2026302
dot/PartitionedCallPartitionedCall.getOnlyPositiveValues/PartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2026850µ
+actualOutputClasses/StatefulPartitionedCallStatefulPartitionedCalldot/PartitionedCall:output:0actualoutputclasses_2026864actualoutputclasses_2026866*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2026863
IdentityIdentity4actualOutputClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp&^ExperimentEmb/StatefulPartitionedCall,^actualOutputClasses/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 2N
%ExperimentEmb/StatefulPartitionedCall%ExperimentEmb/StatefulPartitionedCall2Z
+actualOutputClasses/StatefulPartitionedCall+actualOutputClasses/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ê8


B__inference_model_layer_call_and_return_conditional_losses_2026748
	vec_input%
transform1_2026697:	B!
transform1_2026699:	&
transform2_2026702:
!
transform2_2026704:	&
transform3_2026707:
!
transform3_2026709:	*
batch_normalization_2026712:	*
batch_normalization_2026714:	*
batch_normalization_2026716:	*
batch_normalization_2026718:	&
transform4_2026722:
!
transform4_2026724:	&
transform5_2026727:
!
transform5_2026729:	,
batch_normalization_1_2026732:	,
batch_normalization_1_2026734:	,
batch_normalization_1_2026736:	,
batch_normalization_1_2026738:	+
secondaryclasses_2026742:	&
secondaryclasses_2026744:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢(secondaryClasses/StatefulPartitionedCall¢"transform1/StatefulPartitionedCall¢"transform2/StatefulPartitionedCall¢"transform3/StatefulPartitionedCall¢"transform4/StatefulPartitionedCall¢"transform5/StatefulPartitionedCallÿ
"transform1/StatefulPartitionedCallStatefulPartitionedCall	vec_inputtransform1_2026697transform1_2026699*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2026178¡
"transform2/StatefulPartitionedCallStatefulPartitionedCall+transform1/StatefulPartitionedCall:output:0transform2_2026702transform2_2026704*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform2_layer_call_and_return_conditional_losses_2026195¡
"transform3/StatefulPartitionedCallStatefulPartitionedCall+transform2/StatefulPartitionedCall:output:0transform3_2026707transform3_2026709*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform3_layer_call_and_return_conditional_losses_2026212
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+transform3/StatefulPartitionedCall:output:0batch_normalization_2026712batch_normalization_2026714batch_normalization_2026716batch_normalization_2026718*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2026067ö
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2026418
"transform4/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0transform4_2026722transform4_2026724*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform4_layer_call_and_return_conditional_losses_2026245¡
"transform5/StatefulPartitionedCallStatefulPartitionedCall+transform4/StatefulPartitionedCall:output:0transform5_2026727transform5_2026729*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform5_layer_call_and_return_conditional_losses_2026262
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall+transform5/StatefulPartitionedCall:output:0batch_normalization_1_2026732batch_normalization_1_2026734batch_normalization_1_2026736batch_normalization_1_2026738*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2026149
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2026375·
(secondaryClasses/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0secondaryclasses_2026742secondaryclasses_2026744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2026295
IdentityIdentity1secondaryClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall)^secondaryClasses/StatefulPartitionedCall#^transform1/StatefulPartitionedCall#^transform2/StatefulPartitionedCall#^transform3/StatefulPartitionedCall#^transform4/StatefulPartitionedCall#^transform5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2T
(secondaryClasses/StatefulPartitionedCall(secondaryClasses/StatefulPartitionedCall2H
"transform1/StatefulPartitionedCall"transform1/StatefulPartitionedCall2H
"transform2/StatefulPartitionedCall"transform2/StatefulPartitionedCall2H
"transform3/StatefulPartitionedCall"transform3/StatefulPartitionedCall2H
"transform4/StatefulPartitionedCall"transform4/StatefulPartitionedCall2H
"transform5/StatefulPartitionedCall"transform5/StatefulPartitionedCall:R N
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
#
_user_specified_name	vec_input
¦

ú
G__inference_transform1_layer_call_and_return_conditional_losses_2028129

inputs1
matmul_readvariableop_resource:	B.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	B*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿB: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
ú	
c
D__inference_dropout_layer_call_and_return_conditional_losses_2028276

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®%
í
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2028249

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°¬
À1
#__inference__traced_restore_2028932
file_prefix=
)assignvariableop_experimentemb_embeddings:
²?
-assignvariableop_1_actualoutputclasses_kernel:9
+assignvariableop_2_actualoutputclasses_bias:&
assignvariableop_3_adam_iter:	 (
assignvariableop_4_adam_beta_1: (
assignvariableop_5_adam_beta_2: '
assignvariableop_6_adam_decay: /
%assignvariableop_7_adam_learning_rate: 7
$assignvariableop_8_transform1_kernel:	B1
"assignvariableop_9_transform1_bias:	9
%assignvariableop_10_transform2_kernel:
2
#assignvariableop_11_transform2_bias:	9
%assignvariableop_12_transform3_kernel:
2
#assignvariableop_13_transform3_bias:	<
-assignvariableop_14_batch_normalization_gamma:	;
,assignvariableop_15_batch_normalization_beta:	B
3assignvariableop_16_batch_normalization_moving_mean:	F
7assignvariableop_17_batch_normalization_moving_variance:	9
%assignvariableop_18_transform4_kernel:
2
#assignvariableop_19_transform4_bias:	9
%assignvariableop_20_transform5_kernel:
2
#assignvariableop_21_transform5_bias:	>
/assignvariableop_22_batch_normalization_1_gamma:	=
.assignvariableop_23_batch_normalization_1_beta:	D
5assignvariableop_24_batch_normalization_1_moving_mean:	H
9assignvariableop_25_batch_normalization_1_moving_variance:	>
+assignvariableop_26_secondaryclasses_kernel:	7
)assignvariableop_27_secondaryclasses_bias:#
assignvariableop_28_total: #
assignvariableop_29_count: 1
"assignvariableop_30_true_positives:	È1
"assignvariableop_31_true_negatives:	È2
#assignvariableop_32_false_positives:	È2
#assignvariableop_33_false_negatives:	È3
$assignvariableop_34_true_positives_1:	È3
$assignvariableop_35_true_negatives_1:	È4
%assignvariableop_36_false_positives_1:	È4
%assignvariableop_37_false_negatives_1:	ÈG
3assignvariableop_38_adam_experimentemb_embeddings_m:
²G
5assignvariableop_39_adam_actualoutputclasses_kernel_m:A
3assignvariableop_40_adam_actualoutputclasses_bias_m:?
,assignvariableop_41_adam_transform1_kernel_m:	B9
*assignvariableop_42_adam_transform1_bias_m:	@
,assignvariableop_43_adam_transform2_kernel_m:
9
*assignvariableop_44_adam_transform2_bias_m:	@
,assignvariableop_45_adam_transform3_kernel_m:
9
*assignvariableop_46_adam_transform3_bias_m:	C
4assignvariableop_47_adam_batch_normalization_gamma_m:	B
3assignvariableop_48_adam_batch_normalization_beta_m:	@
,assignvariableop_49_adam_transform4_kernel_m:
9
*assignvariableop_50_adam_transform4_bias_m:	@
,assignvariableop_51_adam_transform5_kernel_m:
9
*assignvariableop_52_adam_transform5_bias_m:	E
6assignvariableop_53_adam_batch_normalization_1_gamma_m:	D
5assignvariableop_54_adam_batch_normalization_1_beta_m:	E
2assignvariableop_55_adam_secondaryclasses_kernel_m:	>
0assignvariableop_56_adam_secondaryclasses_bias_m:G
3assignvariableop_57_adam_experimentemb_embeddings_v:
²G
5assignvariableop_58_adam_actualoutputclasses_kernel_v:A
3assignvariableop_59_adam_actualoutputclasses_bias_v:?
,assignvariableop_60_adam_transform1_kernel_v:	B9
*assignvariableop_61_adam_transform1_bias_v:	@
,assignvariableop_62_adam_transform2_kernel_v:
9
*assignvariableop_63_adam_transform2_bias_v:	@
,assignvariableop_64_adam_transform3_kernel_v:
9
*assignvariableop_65_adam_transform3_bias_v:	C
4assignvariableop_66_adam_batch_normalization_gamma_v:	B
3assignvariableop_67_adam_batch_normalization_beta_v:	@
,assignvariableop_68_adam_transform4_kernel_v:
9
*assignvariableop_69_adam_transform4_bias_v:	@
,assignvariableop_70_adam_transform5_kernel_v:
9
*assignvariableop_71_adam_transform5_bias_v:	E
6assignvariableop_72_adam_batch_normalization_1_gamma_v:	D
5assignvariableop_73_adam_batch_normalization_1_beta_v:	E
2assignvariableop_74_adam_secondaryclasses_kernel_v:	>
0assignvariableop_75_adam_secondaryclasses_bias_v:
identity_77¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_8¢AssignVariableOp_9»$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*á#
value×#BÔ#MB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*¯
value¥B¢MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ¢
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ê
_output_shapes·
´:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*[
dtypesQ
O2M	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp)assignvariableop_experimentemb_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp-assignvariableop_1_actualoutputclasses_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp+assignvariableop_2_actualoutputclasses_biasIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_iterIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_1Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_2Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_decayIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp%assignvariableop_7_adam_learning_rateIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp$assignvariableop_8_transform1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp"assignvariableop_9_transform1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp%assignvariableop_10_transform2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp#assignvariableop_11_transform2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp%assignvariableop_12_transform3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp#assignvariableop_13_transform3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp-assignvariableop_14_batch_normalization_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp,assignvariableop_15_batch_normalization_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_16AssignVariableOp3assignvariableop_16_batch_normalization_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_17AssignVariableOp7assignvariableop_17_batch_normalization_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp%assignvariableop_18_transform4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp#assignvariableop_19_transform4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp%assignvariableop_20_transform5_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp#assignvariableop_21_transform5_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_22AssignVariableOp/assignvariableop_22_batch_normalization_1_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp.assignvariableop_23_batch_normalization_1_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_24AssignVariableOp5assignvariableop_24_batch_normalization_1_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_25AssignVariableOp9assignvariableop_25_batch_normalization_1_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp+assignvariableop_26_secondaryclasses_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp)assignvariableop_27_secondaryclasses_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_totalIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_countIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp"assignvariableop_30_true_positivesIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp"assignvariableop_31_true_negativesIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp#assignvariableop_32_false_positivesIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp#assignvariableop_33_false_negativesIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp$assignvariableop_34_true_positives_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp$assignvariableop_35_true_negatives_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp%assignvariableop_36_false_positives_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp%assignvariableop_37_false_negatives_1Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_experimentemb_embeddings_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_39AssignVariableOp5assignvariableop_39_adam_actualoutputclasses_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_40AssignVariableOp3assignvariableop_40_adam_actualoutputclasses_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp,assignvariableop_41_adam_transform1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp*assignvariableop_42_adam_transform1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp,assignvariableop_43_adam_transform2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp*assignvariableop_44_adam_transform2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp,assignvariableop_45_adam_transform3_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp*assignvariableop_46_adam_transform3_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_batch_normalization_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_48AssignVariableOp3assignvariableop_48_adam_batch_normalization_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_49AssignVariableOp,assignvariableop_49_adam_transform4_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_transform4_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_transform5_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_transform5_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_53AssignVariableOp6assignvariableop_53_adam_batch_normalization_1_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_54AssignVariableOp5assignvariableop_54_adam_batch_normalization_1_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_secondaryclasses_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_56AssignVariableOp0assignvariableop_56_adam_secondaryclasses_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_57AssignVariableOp3assignvariableop_57_adam_experimentemb_embeddings_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_actualoutputclasses_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_59AssignVariableOp3assignvariableop_59_adam_actualoutputclasses_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp,assignvariableop_60_adam_transform1_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_transform1_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp,assignvariableop_62_adam_transform2_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_transform2_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp,assignvariableop_64_adam_transform3_kernel_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_transform3_bias_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_66AssignVariableOp4assignvariableop_66_adam_batch_normalization_gamma_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_67AssignVariableOp3assignvariableop_67_adam_batch_normalization_beta_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp,assignvariableop_68_adam_transform4_kernel_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_transform4_bias_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp,assignvariableop_70_adam_transform5_kernel_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_transform5_bias_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_72AssignVariableOp6assignvariableop_72_adam_batch_normalization_1_gamma_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:¦
AssignVariableOp_73AssignVariableOp5assignvariableop_73_adam_batch_normalization_1_beta_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_74AssignVariableOp2assignvariableop_74_adam_secondaryclasses_kernel_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_75AssignVariableOp0assignvariableop_75_adam_secondaryclasses_bias_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ×
Identity_76Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_77IdentityIdentity_76:output:0^NoOp_1*
T0*
_output_shapes
: Ä
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_77Identity_77:output:0*¯
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
á
µ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2026102

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¡
E
)__inference_dropout_layer_call_fn_2028254

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2026232a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦

ú
G__inference_transform1_layer_call_and_return_conditional_losses_2026178

inputs1
matmul_readvariableop_resource:	B.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	B*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿB: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
°	
ª
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2026773

inputs,
embedding_lookup_2026767:
²
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
embedding_lookupResourceGatherembedding_lookup_2026767Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2026767*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/2026767*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_2026282

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç

'__inference_model_layer_call_fn_2027871

inputs
unknown:	B
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2026552o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
Âj
Ô
B__inference_model_layer_call_and_return_conditional_losses_2027951

inputs<
)transform1_matmul_readvariableop_resource:	B9
*transform1_biasadd_readvariableop_resource:	=
)transform2_matmul_readvariableop_resource:
9
*transform2_biasadd_readvariableop_resource:	=
)transform3_matmul_readvariableop_resource:
9
*transform3_biasadd_readvariableop_resource:	D
5batch_normalization_batchnorm_readvariableop_resource:	H
9batch_normalization_batchnorm_mul_readvariableop_resource:	F
7batch_normalization_batchnorm_readvariableop_1_resource:	F
7batch_normalization_batchnorm_readvariableop_2_resource:	=
)transform4_matmul_readvariableop_resource:
9
*transform4_biasadd_readvariableop_resource:	=
)transform5_matmul_readvariableop_resource:
9
*transform5_biasadd_readvariableop_resource:	F
7batch_normalization_1_batchnorm_readvariableop_resource:	J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	H
9batch_normalization_1_batchnorm_readvariableop_1_resource:	H
9batch_normalization_1_batchnorm_readvariableop_2_resource:	B
/secondaryclasses_matmul_readvariableop_resource:	>
0secondaryclasses_biasadd_readvariableop_resource:
identity¢,batch_normalization/batchnorm/ReadVariableOp¢.batch_normalization/batchnorm/ReadVariableOp_1¢.batch_normalization/batchnorm/ReadVariableOp_2¢0batch_normalization/batchnorm/mul/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢0batch_normalization_1/batchnorm/ReadVariableOp_1¢0batch_normalization_1/batchnorm/ReadVariableOp_2¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢'secondaryClasses/BiasAdd/ReadVariableOp¢&secondaryClasses/MatMul/ReadVariableOp¢!transform1/BiasAdd/ReadVariableOp¢ transform1/MatMul/ReadVariableOp¢!transform2/BiasAdd/ReadVariableOp¢ transform2/MatMul/ReadVariableOp¢!transform3/BiasAdd/ReadVariableOp¢ transform3/MatMul/ReadVariableOp¢!transform4/BiasAdd/ReadVariableOp¢ transform4/MatMul/ReadVariableOp¢!transform5/BiasAdd/ReadVariableOp¢ transform5/MatMul/ReadVariableOp
 transform1/MatMul/ReadVariableOpReadVariableOp)transform1_matmul_readvariableop_resource*
_output_shapes
:	B*
dtype0
transform1/MatMulMatMulinputs(transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform1/BiasAdd/ReadVariableOpReadVariableOp*transform1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform1/BiasAddBiasAddtransform1/MatMul:product:0)transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform1/SeluSelutransform1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 transform2/MatMul/ReadVariableOpReadVariableOp)transform2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
transform2/MatMulMatMultransform1/Selu:activations:0(transform2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform2/BiasAdd/ReadVariableOpReadVariableOp*transform2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform2/BiasAddBiasAddtransform2/MatMul:product:0)transform2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform2/SeluSelutransform2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 transform3/MatMul/ReadVariableOpReadVariableOp)transform3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
transform3/MatMulMatMultransform2/Selu:activations:0(transform3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform3/BiasAdd/ReadVariableOpReadVariableOp*transform3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform3/BiasAddBiasAddtransform3/MatMul:product:0)transform3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform3/SeluSelutransform3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:´
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:§
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0±
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:£
#batch_normalization/batchnorm/mul_1Multransform3/Selu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0¯
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:£
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0¯
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¯
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
dropout/IdentityIdentity'batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 transform4/MatMul/ReadVariableOpReadVariableOp)transform4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
transform4/MatMulMatMuldropout/Identity:output:0(transform4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform4/BiasAdd/ReadVariableOpReadVariableOp*transform4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform4/BiasAddBiasAddtransform4/MatMul:product:0)transform4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform4/SeluSelutransform4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 transform5/MatMul/ReadVariableOpReadVariableOp)transform5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
transform5/MatMulMatMultransform4/Selu:activations:0(transform5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform5/BiasAdd/ReadVariableOpReadVariableOp*transform5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform5/BiasAddBiasAddtransform5/MatMul:product:0)transform5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform5/SeluSelutransform5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:º
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:«
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0·
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:§
%batch_normalization_1/batchnorm/mul_1Multransform5/Selu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0µ
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:§
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0µ
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:µ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
dropout_1/IdentityIdentity)batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&secondaryClasses/MatMul/ReadVariableOpReadVariableOp/secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0 
secondaryClasses/MatMulMatMuldropout_1/Identity:output:0.secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp0secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
secondaryClasses/BiasAddBiasAdd!secondaryClasses/MatMul:product:0/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
secondaryClasses/SoftmaxSoftmax!secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
IdentityIdentity"secondaryClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp1^batch_normalization_1/batchnorm/ReadVariableOp_11^batch_normalization_1/batchnorm/ReadVariableOp_23^batch_normalization_1/batchnorm/mul/ReadVariableOp(^secondaryClasses/BiasAdd/ReadVariableOp'^secondaryClasses/MatMul/ReadVariableOp"^transform1/BiasAdd/ReadVariableOp!^transform1/MatMul/ReadVariableOp"^transform2/BiasAdd/ReadVariableOp!^transform2/MatMul/ReadVariableOp"^transform3/BiasAdd/ReadVariableOp!^transform3/MatMul/ReadVariableOp"^transform4/BiasAdd/ReadVariableOp!^transform4/MatMul/ReadVariableOp"^transform5/BiasAdd/ReadVariableOp!^transform5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 2\
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
!transform1/BiasAdd/ReadVariableOp!transform1/BiasAdd/ReadVariableOp2D
 transform1/MatMul/ReadVariableOp transform1/MatMul/ReadVariableOp2F
!transform2/BiasAdd/ReadVariableOp!transform2/BiasAdd/ReadVariableOp2D
 transform2/MatMul/ReadVariableOp transform2/MatMul/ReadVariableOp2F
!transform3/BiasAdd/ReadVariableOp!transform3/BiasAdd/ReadVariableOp2D
 transform3/MatMul/ReadVariableOp transform3/MatMul/ReadVariableOp2F
!transform4/BiasAdd/ReadVariableOp!transform4/BiasAdd/ReadVariableOp2D
 transform4/MatMul/ReadVariableOp transform4/MatMul/ReadVariableOp2F
!transform5/BiasAdd/ReadVariableOp!transform5/BiasAdd/ReadVariableOp2D
 transform5/MatMul/ReadVariableOp transform5/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
Û
b
D__inference_dropout_layer_call_and_return_conditional_losses_2028264

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

,__inference_transform3_layer_call_fn_2028158

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform3_layer_call_and_return_conditional_losses_2026212p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
d
F__inference_dropout_1_layer_call_and_return_conditional_losses_2028411

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_2028423

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯
E
)__inference_reshape_layer_call_fn_2027758

inputs
identity³
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2026790d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
É
S
7__inference_getOnlyPositiveValues_layer_call_fn_2027776

inputs
identityÁ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2026797d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ¬
Õ
"__inference__wrapped_model_2025996
vec_dateB
.model_1_experimentemb_embedding_lookup_2025892:
²J
7model_1_model_transform1_matmul_readvariableop_resource:	BG
8model_1_model_transform1_biasadd_readvariableop_resource:	K
7model_1_model_transform2_matmul_readvariableop_resource:
G
8model_1_model_transform2_biasadd_readvariableop_resource:	K
7model_1_model_transform3_matmul_readvariableop_resource:
G
8model_1_model_transform3_biasadd_readvariableop_resource:	R
Cmodel_1_model_batch_normalization_batchnorm_readvariableop_resource:	V
Gmodel_1_model_batch_normalization_batchnorm_mul_readvariableop_resource:	T
Emodel_1_model_batch_normalization_batchnorm_readvariableop_1_resource:	T
Emodel_1_model_batch_normalization_batchnorm_readvariableop_2_resource:	K
7model_1_model_transform4_matmul_readvariableop_resource:
G
8model_1_model_transform4_biasadd_readvariableop_resource:	K
7model_1_model_transform5_matmul_readvariableop_resource:
G
8model_1_model_transform5_biasadd_readvariableop_resource:	T
Emodel_1_model_batch_normalization_1_batchnorm_readvariableop_resource:	X
Imodel_1_model_batch_normalization_1_batchnorm_mul_readvariableop_resource:	V
Gmodel_1_model_batch_normalization_1_batchnorm_readvariableop_1_resource:	V
Gmodel_1_model_batch_normalization_1_batchnorm_readvariableop_2_resource:	P
=model_1_model_secondaryclasses_matmul_readvariableop_resource:	L
>model_1_model_secondaryclasses_biasadd_readvariableop_resource:L
:model_1_actualoutputclasses_matmul_readvariableop_resource:I
;model_1_actualoutputclasses_biasadd_readvariableop_resource:
identity¢&model_1/ExperimentEmb/embedding_lookup¢2model_1/actualOutputClasses/BiasAdd/ReadVariableOp¢1model_1/actualOutputClasses/MatMul/ReadVariableOp¢:model_1/model/batch_normalization/batchnorm/ReadVariableOp¢<model_1/model/batch_normalization/batchnorm/ReadVariableOp_1¢<model_1/model/batch_normalization/batchnorm/ReadVariableOp_2¢>model_1/model/batch_normalization/batchnorm/mul/ReadVariableOp¢<model_1/model/batch_normalization_1/batchnorm/ReadVariableOp¢>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1¢>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2¢@model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp¢5model_1/model/secondaryClasses/BiasAdd/ReadVariableOp¢4model_1/model/secondaryClasses/MatMul/ReadVariableOp¢/model_1/model/transform1/BiasAdd/ReadVariableOp¢.model_1/model/transform1/MatMul/ReadVariableOp¢/model_1/model/transform2/BiasAdd/ReadVariableOp¢.model_1/model/transform2/MatMul/ReadVariableOp¢/model_1/model/transform3/BiasAdd/ReadVariableOp¢.model_1/model/transform3/MatMul/ReadVariableOp¢/model_1/model/transform4/BiasAdd/ReadVariableOp¢.model_1/model/transform4/MatMul/ReadVariableOp¢/model_1/model/transform5/BiasAdd/ReadVariableOp¢.model_1/model/transform5/MatMul/ReadVariableOpw
model_1/tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      b
 model_1/tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¥
model_1/tf.split/splitSplitVvec_datemodel_1/tf.split/Const:output:0)model_1/tf.split/split/split_dim:output:0*
T0*

Tlen0*
_output_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿB:ÿÿÿÿÿÿÿÿÿ*
	num_split
model_1/ExperimentEmb/CastCastmodel_1/tf.split/split:output:5*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_1/ExperimentEmb/embedding_lookupResourceGather.model_1_experimentemb_embedding_lookup_2025892model_1/ExperimentEmb/Cast:y:0*
Tindices0*A
_class7
53loc:@model_1/ExperimentEmb/embedding_lookup/2025892*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0æ
/model_1/ExperimentEmb/embedding_lookup/IdentityIdentity/model_1/ExperimentEmb/embedding_lookup:output:0*
T0*A
_class7
53loc:@model_1/ExperimentEmb/embedding_lookup/2025892*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
1model_1/ExperimentEmb/embedding_lookup/Identity_1Identity8model_1/ExperimentEmb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
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
valueB:¡
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
value	B :Ï
model_1/reshape/Reshape/shapePack&model_1/reshape/strided_slice:output:0(model_1/reshape/Reshape/shape/1:output:0(model_1/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:¼
model_1/reshape/ReshapeReshape:model_1/ExperimentEmb/embedding_lookup/Identity_1:output:0&model_1/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"model_1/getOnlyPositiveValues/ReluRelu model_1/reshape/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
.model_1/model/transform1/MatMul/ReadVariableOpReadVariableOp7model_1_model_transform1_matmul_readvariableop_resource*
_output_shapes
:	B*
dtype0µ
model_1/model/transform1/MatMulMatMulmodel_1/tf.split/split:output:46model_1/model/transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/model_1/model/transform1/BiasAdd/ReadVariableOpReadVariableOp8model_1_model_transform1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 model_1/model/transform1/BiasAddBiasAdd)model_1/model/transform1/MatMul:product:07model_1/model/transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/model/transform1/SeluSelu)model_1/model/transform1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
.model_1/model/transform2/MatMul/ReadVariableOpReadVariableOp7model_1_model_transform2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Á
model_1/model/transform2/MatMulMatMul+model_1/model/transform1/Selu:activations:06model_1/model/transform2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/model_1/model/transform2/BiasAdd/ReadVariableOpReadVariableOp8model_1_model_transform2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 model_1/model/transform2/BiasAddBiasAdd)model_1/model/transform2/MatMul:product:07model_1/model/transform2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/model/transform2/SeluSelu)model_1/model/transform2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
.model_1/model/transform3/MatMul/ReadVariableOpReadVariableOp7model_1_model_transform3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Á
model_1/model/transform3/MatMulMatMul+model_1/model/transform2/Selu:activations:06model_1/model/transform3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/model_1/model/transform3/BiasAdd/ReadVariableOpReadVariableOp8model_1_model_transform3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 model_1/model/transform3/BiasAddBiasAdd)model_1/model/transform3/MatMul:product:07model_1/model/transform3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/model/transform3/SeluSelu)model_1/model/transform3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
:model_1/model/batch_normalization/batchnorm/ReadVariableOpReadVariableOpCmodel_1_model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0v
1model_1/model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Þ
/model_1/model/batch_normalization/batchnorm/addAddV2Bmodel_1/model/batch_normalization/batchnorm/ReadVariableOp:value:0:model_1/model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
1model_1/model/batch_normalization/batchnorm/RsqrtRsqrt3model_1/model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:Ã
>model_1/model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpGmodel_1_model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0Û
/model_1/model/batch_normalization/batchnorm/mulMul5model_1/model/batch_normalization/batchnorm/Rsqrt:y:0Fmodel_1/model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Í
1model_1/model/batch_normalization/batchnorm/mul_1Mul+model_1/model/transform3/Selu:activations:03model_1/model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<model_1/model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpEmodel_1_model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ù
1model_1/model/batch_normalization/batchnorm/mul_2MulDmodel_1/model/batch_normalization/batchnorm/ReadVariableOp_1:value:03model_1/model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:¿
<model_1/model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpEmodel_1_model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0Ù
/model_1/model/batch_normalization/batchnorm/subSubDmodel_1/model/batch_normalization/batchnorm/ReadVariableOp_2:value:05model_1/model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ù
1model_1/model/batch_normalization/batchnorm/add_1AddV25model_1/model/batch_normalization/batchnorm/mul_1:z:03model_1/model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/model/dropout/IdentityIdentity5model_1/model/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
.model_1/model/transform4/MatMul/ReadVariableOpReadVariableOp7model_1_model_transform4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0½
model_1/model/transform4/MatMulMatMul'model_1/model/dropout/Identity:output:06model_1/model/transform4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/model_1/model/transform4/BiasAdd/ReadVariableOpReadVariableOp8model_1_model_transform4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 model_1/model/transform4/BiasAddBiasAdd)model_1/model/transform4/MatMul:product:07model_1/model/transform4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/model/transform4/SeluSelu)model_1/model/transform4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
.model_1/model/transform5/MatMul/ReadVariableOpReadVariableOp7model_1_model_transform5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0Á
model_1/model/transform5/MatMulMatMul+model_1/model/transform4/Selu:activations:06model_1/model/transform5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
/model_1/model/transform5/BiasAdd/ReadVariableOpReadVariableOp8model_1_model_transform5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Â
 model_1/model/transform5/BiasAddBiasAdd)model_1/model/transform5/MatMul:product:07model_1/model/transform5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model_1/model/transform5/SeluSelu)model_1/model/transform5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
<model_1/model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOpEmodel_1_model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0x
3model_1/model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:ä
1model_1/model/batch_normalization_1/batchnorm/addAddV2Dmodel_1/model/batch_normalization_1/batchnorm/ReadVariableOp:value:0<model_1/model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
3model_1/model/batch_normalization_1/batchnorm/RsqrtRsqrt5model_1/model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:Ç
@model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpImodel_1_model_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0á
1model_1/model/batch_normalization_1/batchnorm/mulMul7model_1/model/batch_normalization_1/batchnorm/Rsqrt:y:0Hmodel_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ñ
3model_1/model/batch_normalization_1/batchnorm/mul_1Mul+model_1/model/transform5/Selu:activations:05model_1/model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpGmodel_1_model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0ß
3model_1/model/batch_normalization_1/batchnorm/mul_2MulFmodel_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:05model_1/model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ã
>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpGmodel_1_model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0ß
1model_1/model/batch_normalization_1/batchnorm/subSubFmodel_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:07model_1/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:ß
3model_1/model/batch_normalization_1/batchnorm/add_1AddV27model_1/model/batch_normalization_1/batchnorm/mul_1:z:05model_1/model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 model_1/model/dropout_1/IdentityIdentity7model_1/model/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
4model_1/model/secondaryClasses/MatMul/ReadVariableOpReadVariableOp=model_1_model_secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ê
%model_1/model/secondaryClasses/MatMulMatMul)model_1/model/dropout_1/Identity:output:0<model_1/model/secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
5model_1/model/secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp>model_1_model_secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ó
&model_1/model/secondaryClasses/BiasAddBiasAdd/model_1/model/secondaryClasses/MatMul:product:0=model_1/model/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model_1/model/secondaryClasses/SoftmaxSoftmax/model_1/model/secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
model_1/dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :±
model_1/dot/ExpandDims
ExpandDims0model_1/model/secondaryClasses/Softmax:softmax:0#model_1/dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
model_1/dot/MatMulBatchMatMulV20model_1/getOnlyPositiveValues/Relu:activations:0model_1/dot/ExpandDims:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
model_1/dot/ShapeShapemodel_1/dot/MatMul:output:0*
T0*
_output_shapes
:
model_1/dot/SqueezeSqueezemodel_1/dot/MatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ¬
1model_1/actualOutputClasses/MatMul/ReadVariableOpReadVariableOp:model_1_actualoutputclasses_matmul_readvariableop_resource*
_output_shapes

:*
dtype0·
"model_1/actualOutputClasses/MatMulMatMulmodel_1/dot/Squeeze:output:09model_1/actualOutputClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
2model_1/actualOutputClasses/BiasAdd/ReadVariableOpReadVariableOp;model_1_actualoutputclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ê
#model_1/actualOutputClasses/BiasAddBiasAdd,model_1/actualOutputClasses/MatMul:product:0:model_1/actualOutputClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#model_1/actualOutputClasses/SoftmaxSoftmax,model_1/actualOutputClasses/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
IdentityIdentity-model_1/actualOutputClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¶

NoOpNoOp'^model_1/ExperimentEmb/embedding_lookup3^model_1/actualOutputClasses/BiasAdd/ReadVariableOp2^model_1/actualOutputClasses/MatMul/ReadVariableOp;^model_1/model/batch_normalization/batchnorm/ReadVariableOp=^model_1/model/batch_normalization/batchnorm/ReadVariableOp_1=^model_1/model/batch_normalization/batchnorm/ReadVariableOp_2?^model_1/model/batch_normalization/batchnorm/mul/ReadVariableOp=^model_1/model/batch_normalization_1/batchnorm/ReadVariableOp?^model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1?^model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2A^model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp6^model_1/model/secondaryClasses/BiasAdd/ReadVariableOp5^model_1/model/secondaryClasses/MatMul/ReadVariableOp0^model_1/model/transform1/BiasAdd/ReadVariableOp/^model_1/model/transform1/MatMul/ReadVariableOp0^model_1/model/transform2/BiasAdd/ReadVariableOp/^model_1/model/transform2/MatMul/ReadVariableOp0^model_1/model/transform3/BiasAdd/ReadVariableOp/^model_1/model/transform3/MatMul/ReadVariableOp0^model_1/model/transform4/BiasAdd/ReadVariableOp/^model_1/model/transform4/MatMul/ReadVariableOp0^model_1/model/transform5/BiasAdd/ReadVariableOp/^model_1/model/transform5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 2P
&model_1/ExperimentEmb/embedding_lookup&model_1/ExperimentEmb/embedding_lookup2h
2model_1/actualOutputClasses/BiasAdd/ReadVariableOp2model_1/actualOutputClasses/BiasAdd/ReadVariableOp2f
1model_1/actualOutputClasses/MatMul/ReadVariableOp1model_1/actualOutputClasses/MatMul/ReadVariableOp2x
:model_1/model/batch_normalization/batchnorm/ReadVariableOp:model_1/model/batch_normalization/batchnorm/ReadVariableOp2|
<model_1/model/batch_normalization/batchnorm/ReadVariableOp_1<model_1/model/batch_normalization/batchnorm/ReadVariableOp_12|
<model_1/model/batch_normalization/batchnorm/ReadVariableOp_2<model_1/model/batch_normalization/batchnorm/ReadVariableOp_22
>model_1/model/batch_normalization/batchnorm/mul/ReadVariableOp>model_1/model/batch_normalization/batchnorm/mul/ReadVariableOp2|
<model_1/model/batch_normalization_1/batchnorm/ReadVariableOp<model_1/model/batch_normalization_1/batchnorm/ReadVariableOp2
>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_1>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_12
>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_2>model_1/model/batch_normalization_1/batchnorm/ReadVariableOp_22
@model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp@model_1/model/batch_normalization_1/batchnorm/mul/ReadVariableOp2n
5model_1/model/secondaryClasses/BiasAdd/ReadVariableOp5model_1/model/secondaryClasses/BiasAdd/ReadVariableOp2l
4model_1/model/secondaryClasses/MatMul/ReadVariableOp4model_1/model/secondaryClasses/MatMul/ReadVariableOp2b
/model_1/model/transform1/BiasAdd/ReadVariableOp/model_1/model/transform1/BiasAdd/ReadVariableOp2`
.model_1/model/transform1/MatMul/ReadVariableOp.model_1/model/transform1/MatMul/ReadVariableOp2b
/model_1/model/transform2/BiasAdd/ReadVariableOp/model_1/model/transform2/BiasAdd/ReadVariableOp2`
.model_1/model/transform2/MatMul/ReadVariableOp.model_1/model/transform2/MatMul/ReadVariableOp2b
/model_1/model/transform3/BiasAdd/ReadVariableOp/model_1/model/transform3/BiasAdd/ReadVariableOp2`
.model_1/model/transform3/MatMul/ReadVariableOp.model_1/model/transform3/MatMul/ReadVariableOp2b
/model_1/model/transform4/BiasAdd/ReadVariableOp/model_1/model/transform4/BiasAdd/ReadVariableOp2`
.model_1/model/transform4/MatMul/ReadVariableOp.model_1/model/transform4/MatMul/ReadVariableOp2b
/model_1/model/transform5/BiasAdd/ReadVariableOp/model_1/model/transform5/BiasAdd/ReadVariableOp2`
.model_1/model/transform5/MatMul/ReadVariableOp.model_1/model/transform5/MatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
vec_date
ª

û
G__inference_transform2_layer_call_and_return_conditional_losses_2028149

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó
b
)__inference_dropout_layer_call_fn_2028259

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2026418p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à

`
D__inference_reshape_layer_call_and_return_conditional_losses_2027771

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
valueB:Ñ
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
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­

ÿ
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2028443

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_transform4_layer_call_and_return_conditional_losses_2028296

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬


P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2026863

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
Ô
5__inference_batch_normalization_layer_call_fn_2028195

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCallþ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2026067p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°%
ï
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2026149

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°	
ª
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2027753

inputs,
embedding_lookup_2027747:
²
identity¢embedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¾
embedding_lookupResourceGatherembedding_lookup_2027747Cast:y:0*
Tindices0*+
_class!
loc:@embedding_lookup/2027747*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0¤
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*+
_class!
loc:@embedding_lookup/2027747*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
Ö
7__inference_batch_normalization_1_layer_call_fn_2028329

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2026102p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®
Ô
5__inference_batch_normalization_layer_call_fn_2028182

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2026020p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

,__inference_transform4_layer_call_fn_2028285

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform4_layer_call_and_return_conditional_losses_2026245p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð

'__inference_model_layer_call_fn_2026640
	vec_input
unknown:	B
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:
identity¢StatefulPartitionedCallÉ
StatefulPartitionedCallStatefulPartitionedCall	vec_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2026552o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
#
_user_specified_name	vec_input
æ
n
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2026797

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_transform3_layer_call_and_return_conditional_losses_2026212

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
â
%__inference_signature_wrapper_2027360
vec_date
unknown:
²
	unknown_0:	B
	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:


unknown_11:	

unknown_12:


unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:

unknown_20:

unknown_21:
identity¢StatefulPartitionedCallÖ
StatefulPartitionedCallStatefulPartitionedCallvec_dateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_2025996o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
vec_date
ß
³
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2026020

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê
l
@__inference_dot_layer_call_and_return_conditional_losses_2028089
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
:ÿÿÿÿÿÿÿÿÿl
MatMulBatchMatMulV2inputs_0ExpandDims:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:u
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿX
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
6
Ù	
B__inference_model_layer_call_and_return_conditional_losses_2026694
	vec_input%
transform1_2026643:	B!
transform1_2026645:	&
transform2_2026648:
!
transform2_2026650:	&
transform3_2026653:
!
transform3_2026655:	*
batch_normalization_2026658:	*
batch_normalization_2026660:	*
batch_normalization_2026662:	*
batch_normalization_2026664:	&
transform4_2026668:
!
transform4_2026670:	&
transform5_2026673:
!
transform5_2026675:	,
batch_normalization_1_2026678:	,
batch_normalization_1_2026680:	,
batch_normalization_1_2026682:	,
batch_normalization_1_2026684:	+
secondaryclasses_2026688:	&
secondaryclasses_2026690:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢(secondaryClasses/StatefulPartitionedCall¢"transform1/StatefulPartitionedCall¢"transform2/StatefulPartitionedCall¢"transform3/StatefulPartitionedCall¢"transform4/StatefulPartitionedCall¢"transform5/StatefulPartitionedCallÿ
"transform1/StatefulPartitionedCallStatefulPartitionedCall	vec_inputtransform1_2026643transform1_2026645*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2026178¡
"transform2/StatefulPartitionedCallStatefulPartitionedCall+transform1/StatefulPartitionedCall:output:0transform2_2026648transform2_2026650*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform2_layer_call_and_return_conditional_losses_2026195¡
"transform3/StatefulPartitionedCallStatefulPartitionedCall+transform2/StatefulPartitionedCall:output:0transform3_2026653transform3_2026655*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform3_layer_call_and_return_conditional_losses_2026212
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+transform3/StatefulPartitionedCall:output:0batch_normalization_2026658batch_normalization_2026660batch_normalization_2026662batch_normalization_2026664*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2026020æ
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2026232
"transform4/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0transform4_2026668transform4_2026670*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform4_layer_call_and_return_conditional_losses_2026245¡
"transform5/StatefulPartitionedCallStatefulPartitionedCall+transform4/StatefulPartitionedCall:output:0transform5_2026673transform5_2026675*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform5_layer_call_and_return_conditional_losses_2026262
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall+transform5/StatefulPartitionedCall:output:0batch_normalization_1_2026678batch_normalization_1_2026680batch_normalization_1_2026682batch_normalization_1_2026684*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2026102ì
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2026282¯
(secondaryClasses/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0secondaryclasses_2026688secondaryclasses_2026690*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2026295
IdentityIdentity1secondaryClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall)^secondaryClasses/StatefulPartitionedCall#^transform1/StatefulPartitionedCall#^transform2/StatefulPartitionedCall#^transform3/StatefulPartitionedCall#^transform4/StatefulPartitionedCall#^transform5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2T
(secondaryClasses/StatefulPartitionedCall(secondaryClasses/StatefulPartitionedCall2H
"transform1/StatefulPartitionedCall"transform1/StatefulPartitionedCall2H
"transform2/StatefulPartitionedCall"transform2/StatefulPartitionedCall2H
"transform3/StatefulPartitionedCall"transform3/StatefulPartitionedCall2H
"transform4/StatefulPartitionedCall"transform4/StatefulPartitionedCall2H
"transform5/StatefulPartitionedCall"transform5/StatefulPartitionedCall:R N
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
#
_user_specified_name	vec_input
Ó
ä
)__inference_model_1_layer_call_fn_2027462

inputs
unknown:
²
	unknown_0:	B
	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:


unknown_11:	

unknown_12:


unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:

unknown_20:

unknown_21:
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2027073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
ª

û
G__inference_transform3_layer_call_and_return_conditional_losses_2028169

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_transform4_layer_call_and_return_conditional_losses_2026245

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
Ö
7__inference_batch_normalization_1_layer_call_fn_2028342

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2026149p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á8


B__inference_model_layer_call_and_return_conditional_losses_2026552

inputs%
transform1_2026501:	B!
transform1_2026503:	&
transform2_2026506:
!
transform2_2026508:	&
transform3_2026511:
!
transform3_2026513:	*
batch_normalization_2026516:	*
batch_normalization_2026518:	*
batch_normalization_2026520:	*
batch_normalization_2026522:	&
transform4_2026526:
!
transform4_2026528:	&
transform5_2026531:
!
transform5_2026533:	,
batch_normalization_1_2026536:	,
batch_normalization_1_2026538:	,
batch_normalization_1_2026540:	,
batch_normalization_1_2026542:	+
secondaryclasses_2026546:	&
secondaryclasses_2026548:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢(secondaryClasses/StatefulPartitionedCall¢"transform1/StatefulPartitionedCall¢"transform2/StatefulPartitionedCall¢"transform3/StatefulPartitionedCall¢"transform4/StatefulPartitionedCall¢"transform5/StatefulPartitionedCallü
"transform1/StatefulPartitionedCallStatefulPartitionedCallinputstransform1_2026501transform1_2026503*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2026178¡
"transform2/StatefulPartitionedCallStatefulPartitionedCall+transform1/StatefulPartitionedCall:output:0transform2_2026506transform2_2026508*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform2_layer_call_and_return_conditional_losses_2026195¡
"transform3/StatefulPartitionedCallStatefulPartitionedCall+transform2/StatefulPartitionedCall:output:0transform3_2026511transform3_2026513*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform3_layer_call_and_return_conditional_losses_2026212
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+transform3/StatefulPartitionedCall:output:0batch_normalization_2026516batch_normalization_2026518batch_normalization_2026520batch_normalization_2026522*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2026067ö
dropout/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2026418
"transform4/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0transform4_2026526transform4_2026528*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform4_layer_call_and_return_conditional_losses_2026245¡
"transform5/StatefulPartitionedCallStatefulPartitionedCall+transform4/StatefulPartitionedCall:output:0transform5_2026531transform5_2026533*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform5_layer_call_and_return_conditional_losses_2026262
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall+transform5/StatefulPartitionedCall:output:0batch_normalization_1_2026536batch_normalization_1_2026538batch_normalization_1_2026540batch_normalization_1_2026542*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2026149
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2026375·
(secondaryClasses/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0secondaryclasses_2026546secondaryclasses_2026548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2026295
IdentityIdentity1secondaryClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÎ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall)^secondaryClasses/StatefulPartitionedCall#^transform1/StatefulPartitionedCall#^transform2/StatefulPartitionedCall#^transform3/StatefulPartitionedCall#^transform4/StatefulPartitionedCall#^transform5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2T
(secondaryClasses/StatefulPartitionedCall(secondaryClasses/StatefulPartitionedCall2H
"transform1/StatefulPartitionedCall"transform1/StatefulPartitionedCall2H
"transform2/StatefulPartitionedCall"transform2/StatefulPartitionedCall2H
"transform3/StatefulPartitionedCall"transform3/StatefulPartitionedCall2H
"transform4/StatefulPartitionedCall"transform4/StatefulPartitionedCall2H
"transform5/StatefulPartitionedCall"transform5/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
®á
å
D__inference_model_1_layer_call_and_return_conditional_losses_2027736

inputs:
&experimentemb_embedding_lookup_2027590:
²B
/model_transform1_matmul_readvariableop_resource:	B?
0model_transform1_biasadd_readvariableop_resource:	C
/model_transform2_matmul_readvariableop_resource:
?
0model_transform2_biasadd_readvariableop_resource:	C
/model_transform3_matmul_readvariableop_resource:
?
0model_transform3_biasadd_readvariableop_resource:	P
Amodel_batch_normalization_assignmovingavg_readvariableop_resource:	R
Cmodel_batch_normalization_assignmovingavg_1_readvariableop_resource:	N
?model_batch_normalization_batchnorm_mul_readvariableop_resource:	J
;model_batch_normalization_batchnorm_readvariableop_resource:	C
/model_transform4_matmul_readvariableop_resource:
?
0model_transform4_biasadd_readvariableop_resource:	C
/model_transform5_matmul_readvariableop_resource:
?
0model_transform5_biasadd_readvariableop_resource:	R
Cmodel_batch_normalization_1_assignmovingavg_readvariableop_resource:	T
Emodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource:	P
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	L
=model_batch_normalization_1_batchnorm_readvariableop_resource:	H
5model_secondaryclasses_matmul_readvariableop_resource:	D
6model_secondaryclasses_biasadd_readvariableop_resource:D
2actualoutputclasses_matmul_readvariableop_resource:A
3actualoutputclasses_biasadd_readvariableop_resource:
identity¢ExperimentEmb/embedding_lookup¢*actualOutputClasses/BiasAdd/ReadVariableOp¢)actualOutputClasses/MatMul/ReadVariableOp¢)model/batch_normalization/AssignMovingAvg¢8model/batch_normalization/AssignMovingAvg/ReadVariableOp¢+model/batch_normalization/AssignMovingAvg_1¢:model/batch_normalization/AssignMovingAvg_1/ReadVariableOp¢2model/batch_normalization/batchnorm/ReadVariableOp¢6model/batch_normalization/batchnorm/mul/ReadVariableOp¢+model/batch_normalization_1/AssignMovingAvg¢:model/batch_normalization_1/AssignMovingAvg/ReadVariableOp¢-model/batch_normalization_1/AssignMovingAvg_1¢<model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢4model/batch_normalization_1/batchnorm/ReadVariableOp¢8model/batch_normalization_1/batchnorm/mul/ReadVariableOp¢-model/secondaryClasses/BiasAdd/ReadVariableOp¢,model/secondaryClasses/MatMul/ReadVariableOp¢'model/transform1/BiasAdd/ReadVariableOp¢&model/transform1/MatMul/ReadVariableOp¢'model/transform2/BiasAdd/ReadVariableOp¢&model/transform2/MatMul/ReadVariableOp¢'model/transform3/BiasAdd/ReadVariableOp¢&model/transform3/MatMul/ReadVariableOp¢'model/transform4/BiasAdd/ReadVariableOp¢&model/transform4/MatMul/ReadVariableOp¢'model/transform5/BiasAdd/ReadVariableOp¢&model/transform5/MatMul/ReadVariableOpo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
tf.split/splitSplitVinputstf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*
_output_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿB:ÿÿÿÿÿÿÿÿÿ*
	num_splitt
ExperimentEmb/CastCasttf.split/split:output:5*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
ExperimentEmb/embedding_lookupResourceGather&experimentemb_embedding_lookup_2027590ExperimentEmb/Cast:y:0*
Tindices0*9
_class/
-+loc:@ExperimentEmb/embedding_lookup/2027590*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Î
'ExperimentEmb/embedding_lookup/IdentityIdentity'ExperimentEmb/embedding_lookup:output:0*
T0*9
_class/
-+loc:@ExperimentEmb/embedding_lookup/2027590*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)ExperimentEmb/embedding_lookup/Identity_1Identity0ExperimentEmb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
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
valueB:ù
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
value	B :¯
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:¤
reshape/ReshapeReshape2ExperimentEmb/embedding_lookup/Identity_1:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
getOnlyPositiveValues/ReluRelureshape/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform1/MatMul/ReadVariableOpReadVariableOp/model_transform1_matmul_readvariableop_resource*
_output_shapes
:	B*
dtype0
model/transform1/MatMulMatMultf.split/split:output:4.model/transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform1/BiasAdd/ReadVariableOpReadVariableOp0model_transform1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform1/BiasAddBiasAdd!model/transform1/MatMul:product:0/model/transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform1/SeluSelu!model/transform1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform2/MatMul/ReadVariableOpReadVariableOp/model_transform2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
model/transform2/MatMulMatMul#model/transform1/Selu:activations:0.model/transform2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform2/BiasAdd/ReadVariableOpReadVariableOp0model_transform2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform2/BiasAddBiasAdd!model/transform2/MatMul:product:0/model/transform2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform2/SeluSelu!model/transform2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform3/MatMul/ReadVariableOpReadVariableOp/model_transform3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
model/transform3/MatMulMatMul#model/transform2/Selu:activations:0.model/transform3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform3/BiasAdd/ReadVariableOpReadVariableOp0model_transform3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform3/BiasAddBiasAdd!model/transform3/MatMul:product:0/model/transform3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform3/SeluSelu!model/transform3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8model/batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ñ
&model/batch_normalization/moments/meanMean#model/transform3/Selu:activations:0Amodel/batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
.model/batch_normalization/moments/StopGradientStopGradient/model/batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	Ù
3model/batch_normalization/moments/SquaredDifferenceSquaredDifference#model/transform3/Selu:activations:07model/batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
<model/batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: í
*model/batch_normalization/moments/varianceMean7model/batch_normalization/moments/SquaredDifference:z:0Emodel/batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(¢
)model/batch_normalization/moments/SqueezeSqueeze/model/batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 ¨
+model/batch_normalization/moments/Squeeze_1Squeeze3model/batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
/model/batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<·
8model/batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOpAmodel_batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ð
-model/batch_normalization/AssignMovingAvg/subSub@model/batch_normalization/AssignMovingAvg/ReadVariableOp:value:02model/batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ç
-model/batch_normalization/AssignMovingAvg/mulMul1model/batch_normalization/AssignMovingAvg/sub:z:08model/batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
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
×#<»
:model/batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOpCmodel_batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ö
/model/batch_normalization/AssignMovingAvg_1/subSubBmodel/batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:04model/batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Í
/model/batch_normalization/AssignMovingAvg_1/mulMul3model/batch_normalization/AssignMovingAvg_1/sub:z:0:model/batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
+model/batch_normalization/AssignMovingAvg_1AssignSubVariableOpCmodel_batch_normalization_assignmovingavg_1_readvariableop_resource3model/batch_normalization/AssignMovingAvg_1/mul:z:0;^model/batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:À
'model/batch_normalization/batchnorm/addAddV24model/batch_normalization/moments/Squeeze_1:output:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:³
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:µ
)model/batch_normalization/batchnorm/mul_1Mul#model/transform3/Selu:activations:0+model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ·
)model/batch_normalization/batchnorm/mul_2Mul2model/batch_normalization/moments/Squeeze:output:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:«
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0¿
'model/batch_normalization/batchnorm/subSub:model/batch_normalization/batchnorm/ReadVariableOp:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Á
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
model/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?¨
model/dropout/dropout/MulMul-model/batch_normalization/batchnorm/add_1:z:0$model/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
model/dropout/dropout/ShapeShape-model/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:©
2model/dropout/dropout/random_uniform/RandomUniformRandomUniform$model/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0i
$model/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Ñ
"model/dropout/dropout/GreaterEqualGreaterEqual;model/dropout/dropout/random_uniform/RandomUniform:output:0-model/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout/dropout/CastCast&model/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout/dropout/Mul_1Mulmodel/dropout/dropout/Mul:z:0model/dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform4/MatMul/ReadVariableOpReadVariableOp/model_transform4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¥
model/transform4/MatMulMatMulmodel/dropout/dropout/Mul_1:z:0.model/transform4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform4/BiasAdd/ReadVariableOpReadVariableOp0model_transform4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform4/BiasAddBiasAdd!model/transform4/MatMul:product:0/model/transform4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform4/SeluSelu!model/transform4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform5/MatMul/ReadVariableOpReadVariableOp/model_transform5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
model/transform5/MatMulMatMul#model/transform4/Selu:activations:0.model/transform5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform5/BiasAdd/ReadVariableOpReadVariableOp0model_transform5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform5/BiasAddBiasAdd!model/transform5/MatMul:product:0/model/transform5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform5/SeluSelu!model/transform5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:model/batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Õ
(model/batch_normalization_1/moments/meanMean#model/transform5/Selu:activations:0Cmodel/batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
0model/batch_normalization_1/moments/StopGradientStopGradient1model/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	Ý
5model/batch_normalization_1/moments/SquaredDifferenceSquaredDifference#model/transform5/Selu:activations:09model/batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>model/batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ó
,model/batch_normalization_1/moments/varianceMean9model/batch_normalization_1/moments/SquaredDifference:z:0Gmodel/batch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(¦
+model/batch_normalization_1/moments/SqueezeSqueeze1model/batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 ¬
-model/batch_normalization_1/moments/Squeeze_1Squeeze5model/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 v
1model/batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<»
:model/batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOpCmodel_batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ö
/model/batch_normalization_1/AssignMovingAvg/subSubBmodel/batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:04model/batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:Í
/model/batch_normalization_1/AssignMovingAvg/mulMul3model/batch_normalization_1/AssignMovingAvg/sub:z:0:model/batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
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
×#<¿
<model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOpEmodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ü
1model/batch_normalization_1/AssignMovingAvg_1/subSubDmodel/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:06model/batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ó
1model/batch_normalization_1/AssignMovingAvg_1/mulMul5model/batch_normalization_1/AssignMovingAvg_1/sub:z:0<model/batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:¤
-model/batch_normalization_1/AssignMovingAvg_1AssignSubVariableOpEmodel_batch_normalization_1_assignmovingavg_1_readvariableop_resource5model/batch_normalization_1/AssignMovingAvg_1/mul:z:0=^model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Æ
)model/batch_normalization_1/batchnorm/addAddV26model/batch_normalization_1/moments/Squeeze_1:output:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:·
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0É
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¹
+model/batch_normalization_1/batchnorm/mul_1Mul#model/transform5/Selu:activations:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
+model/batch_normalization_1/batchnorm/mul_2Mul4model/batch_normalization_1/moments/Squeeze:output:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:¯
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0Å
)model/batch_normalization_1/batchnorm/subSub<model/batch_normalization_1/batchnorm/ReadVariableOp:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ç
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
model/dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?®
model/dropout_1/dropout/MulMul/model/batch_normalization_1/batchnorm/add_1:z:0&model/dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
model/dropout_1/dropout/ShapeShape/model/batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:­
4model/dropout_1/dropout/random_uniform/RandomUniformRandomUniform&model/dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0k
&model/dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>×
$model/dropout_1/dropout/GreaterEqualGreaterEqual=model/dropout_1/dropout/random_uniform/RandomUniform:output:0/model/dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout_1/dropout/CastCast(model/dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout_1/dropout/Mul_1Mulmodel/dropout_1/dropout/Mul:z:0 model/dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,model/secondaryClasses/MatMul/ReadVariableOpReadVariableOp5model_secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0²
model/secondaryClasses/MatMulMatMul!model/dropout_1/dropout/Mul_1:z:04model/secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-model/secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp6model_secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
model/secondaryClasses/BiasAddBiasAdd'model/secondaryClasses/MatMul:product:05model/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/secondaryClasses/SoftmaxSoftmax'model/secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot/ExpandDims
ExpandDims(model/secondaryClasses/Softmax:softmax:0dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dot/MatMulBatchMatMulV2(getOnlyPositiveValues/Relu:activations:0dot/ExpandDims:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
	dot/ShapeShapedot/MatMul:output:0*
T0*
_output_shapes
:}
dot/SqueezeSqueezedot/MatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ
)actualOutputClasses/MatMul/ReadVariableOpReadVariableOp2actualoutputclasses_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
actualOutputClasses/MatMulMatMuldot/Squeeze:output:01actualOutputClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*actualOutputClasses/BiasAdd/ReadVariableOpReadVariableOp3actualoutputclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
actualOutputClasses/BiasAddBiasAdd$actualOutputClasses/MatMul:product:02actualOutputClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
actualOutputClasses/SoftmaxSoftmax$actualOutputClasses/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%actualOutputClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ

NoOpNoOp^ExperimentEmb/embedding_lookup+^actualOutputClasses/BiasAdd/ReadVariableOp*^actualOutputClasses/MatMul/ReadVariableOp*^model/batch_normalization/AssignMovingAvg9^model/batch_normalization/AssignMovingAvg/ReadVariableOp,^model/batch_normalization/AssignMovingAvg_1;^model/batch_normalization/AssignMovingAvg_1/ReadVariableOp3^model/batch_normalization/batchnorm/ReadVariableOp7^model/batch_normalization/batchnorm/mul/ReadVariableOp,^model/batch_normalization_1/AssignMovingAvg;^model/batch_normalization_1/AssignMovingAvg/ReadVariableOp.^model/batch_normalization_1/AssignMovingAvg_1=^model/batch_normalization_1/AssignMovingAvg_1/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp9^model/batch_normalization_1/batchnorm/mul/ReadVariableOp.^model/secondaryClasses/BiasAdd/ReadVariableOp-^model/secondaryClasses/MatMul/ReadVariableOp(^model/transform1/BiasAdd/ReadVariableOp'^model/transform1/MatMul/ReadVariableOp(^model/transform2/BiasAdd/ReadVariableOp'^model/transform2/MatMul/ReadVariableOp(^model/transform3/BiasAdd/ReadVariableOp'^model/transform3/MatMul/ReadVariableOp(^model/transform4/BiasAdd/ReadVariableOp'^model/transform4/MatMul/ReadVariableOp(^model/transform5/BiasAdd/ReadVariableOp'^model/transform5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 2@
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
'model/transform1/BiasAdd/ReadVariableOp'model/transform1/BiasAdd/ReadVariableOp2P
&model/transform1/MatMul/ReadVariableOp&model/transform1/MatMul/ReadVariableOp2R
'model/transform2/BiasAdd/ReadVariableOp'model/transform2/BiasAdd/ReadVariableOp2P
&model/transform2/MatMul/ReadVariableOp&model/transform2/MatMul/ReadVariableOp2R
'model/transform3/BiasAdd/ReadVariableOp'model/transform3/BiasAdd/ReadVariableOp2P
&model/transform3/MatMul/ReadVariableOp&model/transform3/MatMul/ReadVariableOp2R
'model/transform4/BiasAdd/ReadVariableOp'model/transform4/BiasAdd/ReadVariableOp2P
&model/transform4/MatMul/ReadVariableOp&model/transform4/MatMul/ReadVariableOp2R
'model/transform5/BiasAdd/ReadVariableOp'model/transform5/BiasAdd/ReadVariableOp2P
&model/transform5/MatMul/ReadVariableOp&model/transform5/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
Ë

'__inference_model_layer_call_fn_2027826

inputs
unknown:	B
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:
identity¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2026302o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
ª

û
G__inference_transform5_layer_call_and_return_conditional_losses_2028316

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
&
Ó
D__inference_model_1_layer_call_and_return_conditional_losses_2027301
vec_date)
experimentemb_2027248:
² 
model_2027253:	B
model_2027255:	!
model_2027257:

model_2027259:	!
model_2027261:

model_2027263:	
model_2027265:	
model_2027267:	
model_2027269:	
model_2027271:	!
model_2027273:

model_2027275:	!
model_2027277:

model_2027279:	
model_2027281:	
model_2027283:	
model_2027285:	
model_2027287:	 
model_2027289:	
model_2027291:-
actualoutputclasses_2027295:)
actualoutputclasses_2027297:
identity¢%ExperimentEmb/StatefulPartitionedCall¢+actualOutputClasses/StatefulPartitionedCall¢model/StatefulPartitionedCallo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
tf.split/splitSplitVvec_datetf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*
_output_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿB:ÿÿÿÿÿÿÿÿÿ*
	num_split
%ExperimentEmb/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:5experimentemb_2027248*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2026773ã
reshape/PartitionedCallPartitionedCall.ExperimentEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2026790ñ
%getOnlyPositiveValues/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2026797¦
model/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:4model_2027253model_2027255model_2027257model_2027259model_2027261model_2027263model_2027265model_2027267model_2027269model_2027271model_2027273model_2027275model_2027277model_2027279model_2027281model_2027283model_2027285model_2027287model_2027289model_2027291* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2026552
dot/PartitionedCallPartitionedCall.getOnlyPositiveValues/PartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2026850µ
+actualOutputClasses/StatefulPartitionedCallStatefulPartitionedCalldot/PartitionedCall:output:0actualoutputclasses_2027295actualoutputclasses_2027297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2026863
IdentityIdentity4actualOutputClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp&^ExperimentEmb/StatefulPartitionedCall,^actualOutputClasses/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 2N
%ExperimentEmb/StatefulPartitionedCall%ExperimentEmb/StatefulPartitionedCall2Z
+actualOutputClasses/StatefulPartitionedCall+actualOutputClasses/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
vec_date
Ù
æ
)__inference_model_1_layer_call_fn_2027173
vec_date
unknown:
²
	unknown_0:	B
	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:


unknown_11:	

unknown_12:


unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:

unknown_20:

unknown_21:
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallvec_dateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2027073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
vec_date
&
Ñ
D__inference_model_1_layer_call_and_return_conditional_losses_2027073

inputs)
experimentemb_2027020:
² 
model_2027025:	B
model_2027027:	!
model_2027029:

model_2027031:	!
model_2027033:

model_2027035:	
model_2027037:	
model_2027039:	
model_2027041:	
model_2027043:	!
model_2027045:

model_2027047:	!
model_2027049:

model_2027051:	
model_2027053:	
model_2027055:	
model_2027057:	
model_2027059:	 
model_2027061:	
model_2027063:-
actualoutputclasses_2027067:)
actualoutputclasses_2027069:
identity¢%ExperimentEmb/StatefulPartitionedCall¢+actualOutputClasses/StatefulPartitionedCall¢model/StatefulPartitionedCallo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
tf.split/splitSplitVinputstf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*
_output_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿB:ÿÿÿÿÿÿÿÿÿ*
	num_split
%ExperimentEmb/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:5experimentemb_2027020*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2026773ã
reshape/PartitionedCallPartitionedCall.ExperimentEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2026790ñ
%getOnlyPositiveValues/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2026797¦
model/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:4model_2027025model_2027027model_2027029model_2027031model_2027033model_2027035model_2027037model_2027039model_2027041model_2027043model_2027045model_2027047model_2027049model_2027051model_2027053model_2027055model_2027057model_2027059model_2027061model_2027063* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2026552
dot/PartitionedCallPartitionedCall.getOnlyPositiveValues/PartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2026850µ
+actualOutputClasses/StatefulPartitionedCallStatefulPartitionedCalldot/PartitionedCall:output:0actualoutputclasses_2027067actualoutputclasses_2027069*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2026863
IdentityIdentity4actualOutputClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp&^ExperimentEmb/StatefulPartitionedCall,^actualOutputClasses/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 2N
%ExperimentEmb/StatefulPartitionedCall%ExperimentEmb/StatefulPartitionedCall2Z
+actualOutputClasses/StatefulPartitionedCall+actualOutputClasses/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
¬


P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2028109

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì

,__inference_transform1_layer_call_fn_2028118

inputs
unknown:	B
	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2026178p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿB: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
¥
G
+__inference_dropout_1_layer_call_fn_2028401

inputs
identity²
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2026282a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
æ
)__inference_model_1_layer_call_fn_2026919
vec_date
unknown:
²
	unknown_0:	B
	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:


unknown_11:	

unknown_12:


unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:

unknown_20:

unknown_21:
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallvec_dateunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2026870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
vec_date
Ï

,__inference_transform2_layer_call_fn_2028138

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform2_layer_call_and_return_conditional_losses_2026195p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 
"
 __inference__traced_save_2028694
file_prefix7
3savev2_experimentemb_embeddings_read_readvariableop9
5savev2_actualoutputclasses_kernel_read_readvariableop7
3savev2_actualoutputclasses_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop0
,savev2_transform1_kernel_read_readvariableop.
*savev2_transform1_bias_read_readvariableop0
,savev2_transform2_kernel_read_readvariableop.
*savev2_transform2_bias_read_readvariableop0
,savev2_transform3_kernel_read_readvariableop.
*savev2_transform3_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop0
,savev2_transform4_kernel_read_readvariableop.
*savev2_transform4_bias_read_readvariableop0
,savev2_transform5_kernel_read_readvariableop.
*savev2_transform5_bias_read_readvariableop:
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
3savev2_adam_transform1_kernel_m_read_readvariableop5
1savev2_adam_transform1_bias_m_read_readvariableop7
3savev2_adam_transform2_kernel_m_read_readvariableop5
1savev2_adam_transform2_bias_m_read_readvariableop7
3savev2_adam_transform3_kernel_m_read_readvariableop5
1savev2_adam_transform3_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop7
3savev2_adam_transform4_kernel_m_read_readvariableop5
1savev2_adam_transform4_bias_m_read_readvariableop7
3savev2_adam_transform5_kernel_m_read_readvariableop5
1savev2_adam_transform5_bias_m_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_m_read_readvariableop=
9savev2_adam_secondaryclasses_kernel_m_read_readvariableop;
7savev2_adam_secondaryclasses_bias_m_read_readvariableop>
:savev2_adam_experimentemb_embeddings_v_read_readvariableop@
<savev2_adam_actualoutputclasses_kernel_v_read_readvariableop>
:savev2_adam_actualoutputclasses_bias_v_read_readvariableop7
3savev2_adam_transform1_kernel_v_read_readvariableop5
1savev2_adam_transform1_bias_v_read_readvariableop7
3savev2_adam_transform2_kernel_v_read_readvariableop5
1savev2_adam_transform2_bias_v_read_readvariableop7
3savev2_adam_transform3_kernel_v_read_readvariableop5
1savev2_adam_transform3_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop7
3savev2_adam_transform4_kernel_v_read_readvariableop5
1savev2_adam_transform4_bias_v_read_readvariableop7
3savev2_adam_transform5_kernel_v_read_readvariableop5
1savev2_adam_transform5_bias_v_read_readvariableopA
=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop@
<savev2_adam_batch_normalization_1_beta_v_read_readvariableop=
9savev2_adam_secondaryclasses_kernel_v_read_readvariableop;
7savev2_adam_secondaryclasses_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ¸$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*á#
value×#BÔ#MB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBVlayer_with_weights-0/embeddings/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:M*
dtype0*¯
value¥B¢MB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ß 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_experimentemb_embeddings_read_readvariableop5savev2_actualoutputclasses_kernel_read_readvariableop3savev2_actualoutputclasses_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop,savev2_transform1_kernel_read_readvariableop*savev2_transform1_bias_read_readvariableop,savev2_transform2_kernel_read_readvariableop*savev2_transform2_bias_read_readvariableop,savev2_transform3_kernel_read_readvariableop*savev2_transform3_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop,savev2_transform4_kernel_read_readvariableop*savev2_transform4_bias_read_readvariableop,savev2_transform5_kernel_read_readvariableop*savev2_transform5_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop2savev2_secondaryclasses_kernel_read_readvariableop0savev2_secondaryclasses_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_1_read_readvariableop+savev2_true_negatives_1_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop:savev2_adam_experimentemb_embeddings_m_read_readvariableop<savev2_adam_actualoutputclasses_kernel_m_read_readvariableop:savev2_adam_actualoutputclasses_bias_m_read_readvariableop3savev2_adam_transform1_kernel_m_read_readvariableop1savev2_adam_transform1_bias_m_read_readvariableop3savev2_adam_transform2_kernel_m_read_readvariableop1savev2_adam_transform2_bias_m_read_readvariableop3savev2_adam_transform3_kernel_m_read_readvariableop1savev2_adam_transform3_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop3savev2_adam_transform4_kernel_m_read_readvariableop1savev2_adam_transform4_bias_m_read_readvariableop3savev2_adam_transform5_kernel_m_read_readvariableop1savev2_adam_transform5_bias_m_read_readvariableop=savev2_adam_batch_normalization_1_gamma_m_read_readvariableop<savev2_adam_batch_normalization_1_beta_m_read_readvariableop9savev2_adam_secondaryclasses_kernel_m_read_readvariableop7savev2_adam_secondaryclasses_bias_m_read_readvariableop:savev2_adam_experimentemb_embeddings_v_read_readvariableop<savev2_adam_actualoutputclasses_kernel_v_read_readvariableop:savev2_adam_actualoutputclasses_bias_v_read_readvariableop3savev2_adam_transform1_kernel_v_read_readvariableop1savev2_adam_transform1_bias_v_read_readvariableop3savev2_adam_transform2_kernel_v_read_readvariableop1savev2_adam_transform2_bias_v_read_readvariableop3savev2_adam_transform3_kernel_v_read_readvariableop1savev2_adam_transform3_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop3savev2_adam_transform4_kernel_v_read_readvariableop1savev2_adam_transform4_bias_v_read_readvariableop3savev2_adam_transform5_kernel_v_read_readvariableop1savev2_adam_transform5_bias_v_read_readvariableop=savev2_adam_batch_normalization_1_gamma_v_read_readvariableop<savev2_adam_batch_normalization_1_beta_v_read_readvariableop9savev2_adam_secondaryclasses_kernel_v_read_readvariableop7savev2_adam_secondaryclasses_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *[
dtypesQ
O2M	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*ð
_input_shapesÞ
Û: :
²::: : : : : :	B::
::
::::::
::
::::::	:: : :È:È:È:È:È:È:È:È:
²:::	B::
::
::::
::
::::	::
²:::	B::
::
::::
::
::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
²:$ 

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
:	B:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:È:! 

_output_shapes	
:È:!!

_output_shapes	
:È:!"

_output_shapes	
:È:!#

_output_shapes	
:È:!$

_output_shapes	
:È:!%

_output_shapes	
:È:!&

_output_shapes	
:È:&'"
 
_output_shapes
:
²:$( 

_output_shapes

:: )

_output_shapes
::%*!

_output_shapes
:	B:!+

_output_shapes	
::&,"
 
_output_shapes
:
:!-

_output_shapes	
::&."
 
_output_shapes
:
:!/

_output_shapes	
::!0

_output_shapes	
::!1

_output_shapes	
::&2"
 
_output_shapes
:
:!3

_output_shapes	
::&4"
 
_output_shapes
:
:!5

_output_shapes	
::!6

_output_shapes	
::!7

_output_shapes	
::%8!

_output_shapes
:	: 9

_output_shapes
::&:"
 
_output_shapes
:
²:$; 

_output_shapes

:: <

_output_shapes
::%=!

_output_shapes
:	B:!>

_output_shapes	
::&?"
 
_output_shapes
:
:!@

_output_shapes	
::&A"
 
_output_shapes
:
:!B

_output_shapes	
::!C

_output_shapes	
::!D

_output_shapes	
::&E"
 
_output_shapes
:
:!F

_output_shapes	
::&G"
 
_output_shapes
:
:!H

_output_shapes	
::!I

_output_shapes	
::!J

_output_shapes	
::%K!

_output_shapes
:	: L

_output_shapes
::M

_output_shapes
: 
ü	
e
F__inference_dropout_1_layer_call_and_return_conditional_losses_2026375

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®%
í
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2026067

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
à

`
D__inference_reshape_layer_call_and_return_conditional_losses_2026790

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
valueB:Ñ
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
value	B :
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:T P
,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ú
¢
5__inference_actualOutputClasses_layer_call_fn_2028098

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2026863o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
 
2__inference_secondaryClasses_layer_call_fn_2028432

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallâ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2026295o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´¬

B__inference_model_layer_call_and_return_conditional_losses_2028073

inputs<
)transform1_matmul_readvariableop_resource:	B9
*transform1_biasadd_readvariableop_resource:	=
)transform2_matmul_readvariableop_resource:
9
*transform2_biasadd_readvariableop_resource:	=
)transform3_matmul_readvariableop_resource:
9
*transform3_biasadd_readvariableop_resource:	J
;batch_normalization_assignmovingavg_readvariableop_resource:	L
=batch_normalization_assignmovingavg_1_readvariableop_resource:	H
9batch_normalization_batchnorm_mul_readvariableop_resource:	D
5batch_normalization_batchnorm_readvariableop_resource:	=
)transform4_matmul_readvariableop_resource:
9
*transform4_biasadd_readvariableop_resource:	=
)transform5_matmul_readvariableop_resource:
9
*transform5_biasadd_readvariableop_resource:	L
=batch_normalization_1_assignmovingavg_readvariableop_resource:	N
?batch_normalization_1_assignmovingavg_1_readvariableop_resource:	J
;batch_normalization_1_batchnorm_mul_readvariableop_resource:	F
7batch_normalization_1_batchnorm_readvariableop_resource:	B
/secondaryclasses_matmul_readvariableop_resource:	>
0secondaryclasses_biasadd_readvariableop_resource:
identity¢#batch_normalization/AssignMovingAvg¢2batch_normalization/AssignMovingAvg/ReadVariableOp¢%batch_normalization/AssignMovingAvg_1¢4batch_normalization/AssignMovingAvg_1/ReadVariableOp¢,batch_normalization/batchnorm/ReadVariableOp¢0batch_normalization/batchnorm/mul/ReadVariableOp¢%batch_normalization_1/AssignMovingAvg¢4batch_normalization_1/AssignMovingAvg/ReadVariableOp¢'batch_normalization_1/AssignMovingAvg_1¢6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp¢.batch_normalization_1/batchnorm/ReadVariableOp¢2batch_normalization_1/batchnorm/mul/ReadVariableOp¢'secondaryClasses/BiasAdd/ReadVariableOp¢&secondaryClasses/MatMul/ReadVariableOp¢!transform1/BiasAdd/ReadVariableOp¢ transform1/MatMul/ReadVariableOp¢!transform2/BiasAdd/ReadVariableOp¢ transform2/MatMul/ReadVariableOp¢!transform3/BiasAdd/ReadVariableOp¢ transform3/MatMul/ReadVariableOp¢!transform4/BiasAdd/ReadVariableOp¢ transform4/MatMul/ReadVariableOp¢!transform5/BiasAdd/ReadVariableOp¢ transform5/MatMul/ReadVariableOp
 transform1/MatMul/ReadVariableOpReadVariableOp)transform1_matmul_readvariableop_resource*
_output_shapes
:	B*
dtype0
transform1/MatMulMatMulinputs(transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform1/BiasAdd/ReadVariableOpReadVariableOp*transform1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform1/BiasAddBiasAddtransform1/MatMul:product:0)transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform1/SeluSelutransform1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 transform2/MatMul/ReadVariableOpReadVariableOp)transform2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
transform2/MatMulMatMultransform1/Selu:activations:0(transform2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform2/BiasAdd/ReadVariableOpReadVariableOp*transform2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform2/BiasAddBiasAddtransform2/MatMul:product:0)transform2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform2/SeluSelutransform2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 transform3/MatMul/ReadVariableOpReadVariableOp)transform3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
transform3/MatMulMatMultransform2/Selu:activations:0(transform3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform3/BiasAdd/ReadVariableOpReadVariableOp*transform3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform3/BiasAddBiasAddtransform3/MatMul:product:0)transform3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform3/SeluSelutransform3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ¿
 batch_normalization/moments/meanMeantransform3/Selu:activations:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:	Ç
-batch_normalization/moments/SquaredDifferenceSquaredDifferencetransform3/Selu:activations:01batch_normalization/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Û
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 n
)batch_normalization/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<«
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp;batch_normalization_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0¾
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*
_output_shapes	
:µ
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ü
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
×#<¯
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:»
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
%batch_normalization/AssignMovingAvg_1AssignSubVariableOp=batch_normalization_assignmovingavg_1_readvariableop_resource-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0h
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:®
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:y
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:§
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0±
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:£
#batch_normalization/batchnorm/mul_1Multransform3/Selu:activations:0%batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0­
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:¯
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout/dropout/MulMul'batch_normalization/batchnorm/add_1:z:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
dropout/dropout/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>¿
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 transform4/MatMul/ReadVariableOpReadVariableOp)transform4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
transform4/MatMulMatMuldropout/dropout/Mul_1:z:0(transform4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform4/BiasAdd/ReadVariableOpReadVariableOp*transform4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform4/BiasAddBiasAddtransform4/MatMul:product:0)transform4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform4/SeluSelutransform4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 transform5/MatMul/ReadVariableOpReadVariableOp)transform5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
transform5/MatMulMatMultransform4/Selu:activations:0(transform5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!transform5/BiasAdd/ReadVariableOpReadVariableOp*transform5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
transform5/BiasAddBiasAddtransform5/MatMul:product:0)transform5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
transform5/SeluSelutransform5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ã
"batch_normalization_1/moments/meanMeantransform5/Selu:activations:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
:	Ë
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferencetransform5/Selu:activations:03batch_normalization_1/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: á
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
  
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 p
+batch_normalization_1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<¯
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp=batch_normalization_1_assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0Ä
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*
_output_shapes	
:»
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:
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
×#<³
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0Ê
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Á
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:
'batch_normalization_1/AssignMovingAvg_1AssignSubVariableOp?batch_normalization_1_assignmovingavg_1_readvariableop_resource/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0j
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:´
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:}
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:«
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0·
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:§
%batch_normalization_1/batchnorm/mul_1Multransform5/Selu:activations:0'batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:£
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0³
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:µ
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?
dropout_1/dropout/MulMul)batch_normalization_1/batchnorm/add_1:z:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout_1/dropout/ShapeShape)batch_normalization_1/batchnorm/add_1:z:0*
T0*
_output_shapes
:¡
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>Å
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&secondaryClasses/MatMul/ReadVariableOpReadVariableOp/secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0 
secondaryClasses/MatMulMatMuldropout_1/dropout/Mul_1:z:0.secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp0secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
secondaryClasses/BiasAddBiasAdd!secondaryClasses/MatMul:product:0/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
secondaryClasses/SoftmaxSoftmax!secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
IdentityIdentity"secondaryClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp$^batch_normalization/AssignMovingAvg3^batch_normalization/AssignMovingAvg/ReadVariableOp&^batch_normalization/AssignMovingAvg_15^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp&^batch_normalization_1/AssignMovingAvg5^batch_normalization_1/AssignMovingAvg/ReadVariableOp(^batch_normalization_1/AssignMovingAvg_17^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp/^batch_normalization_1/batchnorm/ReadVariableOp3^batch_normalization_1/batchnorm/mul/ReadVariableOp(^secondaryClasses/BiasAdd/ReadVariableOp'^secondaryClasses/MatMul/ReadVariableOp"^transform1/BiasAdd/ReadVariableOp!^transform1/MatMul/ReadVariableOp"^transform2/BiasAdd/ReadVariableOp!^transform2/MatMul/ReadVariableOp"^transform3/BiasAdd/ReadVariableOp!^transform3/MatMul/ReadVariableOp"^transform4/BiasAdd/ReadVariableOp!^transform4/MatMul/ReadVariableOp"^transform5/BiasAdd/ReadVariableOp!^transform5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 2J
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
!transform1/BiasAdd/ReadVariableOp!transform1/BiasAdd/ReadVariableOp2D
 transform1/MatMul/ReadVariableOp transform1/MatMul/ReadVariableOp2F
!transform2/BiasAdd/ReadVariableOp!transform2/BiasAdd/ReadVariableOp2D
 transform2/MatMul/ReadVariableOp transform2/MatMul/ReadVariableOp2F
!transform3/BiasAdd/ReadVariableOp!transform3/BiasAdd/ReadVariableOp2D
 transform3/MatMul/ReadVariableOp transform3/MatMul/ReadVariableOp2F
!transform4/BiasAdd/ReadVariableOp!transform4/BiasAdd/ReadVariableOp2D
 transform4/MatMul/ReadVariableOp transform4/MatMul/ReadVariableOp2F
!transform5/BiasAdd/ReadVariableOp!transform5/BiasAdd/ReadVariableOp2D
 transform5/MatMul/ReadVariableOp transform5/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs
â
j
@__inference_dot_layer_call_and_return_conditional_losses_2026850

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
:ÿÿÿÿÿÿÿÿÿj
MatMulBatchMatMulV2inputsExpandDims:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:u
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿX
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

'__inference_model_layer_call_fn_2026345
	vec_input
unknown:	B
	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:

	unknown_4:	
	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:


unknown_10:	

unknown_11:


unknown_12:	

unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:
identity¢StatefulPartitionedCallÍ
StatefulPartitionedCallStatefulPartitionedCall	vec_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2026302o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
#
_user_specified_name	vec_input
ß
³
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2028215

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­

ÿ
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2026295

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°%
ï
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2028396

inputs6
'assignmovingavg_readvariableop_resource:	8
)assignmovingavg_1_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	0
!batchnorm_readvariableop_resource:	
identity¢AssignMovingAvg¢AssignMovingAvg/ReadVariableOp¢AssignMovingAvg_1¢ AssignMovingAvg_1/ReadVariableOp¢batchnorm/ReadVariableOp¢batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
×#<
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:¬
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
×#<
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:*
dtype0
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:´
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_transform5_layer_call_and_return_conditional_losses_2026262

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
×
ä
)__inference_model_1_layer_call_fn_2027411

inputs
unknown:
²
	unknown_0:	B
	unknown_1:	
	unknown_2:

	unknown_3:	
	unknown_4:

	unknown_5:	
	unknown_6:	
	unknown_7:	
	unknown_8:	
	unknown_9:	

unknown_10:


unknown_11:	

unknown_12:


unknown_13:	

unknown_14:	

unknown_15:	

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:

unknown_20:

unknown_21:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21*#
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*9
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_model_1_layer_call_and_return_conditional_losses_2026870o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
Û
b
D__inference_dropout_layer_call_and_return_conditional_losses_2026232

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
d
+__inference_dropout_1_layer_call_fn_2028406

inputs
identity¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2026375p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ú	
c
D__inference_dropout_layer_call_and_return_conditional_losses_2026418

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *nÛ¶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *>§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï

,__inference_transform5_layer_call_fn_2028305

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform5_layer_call_and_return_conditional_losses_2026262p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
n
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2027781

inputs
identityJ
ReluReluinputs*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
IdentityIdentityRelu:activations:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²

/__inference_ExperimentEmb_layer_call_fn_2027743

inputs
unknown:
²
identity¢StatefulPartitionedCall×
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2026773t
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

û
G__inference_transform2_layer_call_and_return_conditional_losses_2026195

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentitySelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
&
Ó
D__inference_model_1_layer_call_and_return_conditional_losses_2027237
vec_date)
experimentemb_2027184:
² 
model_2027189:	B
model_2027191:	!
model_2027193:

model_2027195:	!
model_2027197:

model_2027199:	
model_2027201:	
model_2027203:	
model_2027205:	
model_2027207:	!
model_2027209:

model_2027211:	!
model_2027213:

model_2027215:	
model_2027217:	
model_2027219:	
model_2027221:	
model_2027223:	 
model_2027225:	
model_2027227:-
actualoutputclasses_2027231:)
actualoutputclasses_2027233:
identity¢%ExperimentEmb/StatefulPartitionedCall¢+actualOutputClasses/StatefulPartitionedCall¢model/StatefulPartitionedCallo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
tf.split/splitSplitVvec_datetf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*
_output_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿB:ÿÿÿÿÿÿÿÿÿ*
	num_split
%ExperimentEmb/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:5experimentemb_2027184*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2026773ã
reshape/PartitionedCallPartitionedCall.ExperimentEmb/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2026790ñ
%getOnlyPositiveValues/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2026797ª
model/StatefulPartitionedCallStatefulPartitionedCalltf.split/split:output:4model_2027189model_2027191model_2027193model_2027195model_2027197model_2027199model_2027201model_2027203model_2027205model_2027207model_2027209model_2027211model_2027213model_2027215model_2027217model_2027219model_2027221model_2027223model_2027225model_2027227* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2026302
dot/PartitionedCallPartitionedCall.getOnlyPositiveValues/PartitionedCall:output:0&model/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2026850µ
+actualOutputClasses/StatefulPartitionedCallStatefulPartitionedCalldot/PartitionedCall:output:0actualoutputclasses_2027231actualoutputclasses_2027233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2026863
IdentityIdentity4actualOutputClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
NoOpNoOp&^ExperimentEmb/StatefulPartitionedCall,^actualOutputClasses/StatefulPartitionedCall^model/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 2N
%ExperimentEmb/StatefulPartitionedCall%ExperimentEmb/StatefulPartitionedCall2Z
+actualOutputClasses/StatefulPartitionedCall+actualOutputClasses/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
"
_user_specified_name
vec_date
á
µ
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2028362

inputs0
!batchnorm_readvariableop_resource:	4
%batchnorm_mul_readvariableop_resource:	2
#batchnorm_readvariableop_1_resource:	2
#batchnorm_readvariableop_2_resource:	
identity¢batchnorm/ReadVariableOp¢batchnorm/ReadVariableOp_1¢batchnorm/ReadVariableOp_2¢batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

D__inference_model_1_layer_call_and_return_conditional_losses_2027578

inputs:
&experimentemb_embedding_lookup_2027474:
²B
/model_transform1_matmul_readvariableop_resource:	B?
0model_transform1_biasadd_readvariableop_resource:	C
/model_transform2_matmul_readvariableop_resource:
?
0model_transform2_biasadd_readvariableop_resource:	C
/model_transform3_matmul_readvariableop_resource:
?
0model_transform3_biasadd_readvariableop_resource:	J
;model_batch_normalization_batchnorm_readvariableop_resource:	N
?model_batch_normalization_batchnorm_mul_readvariableop_resource:	L
=model_batch_normalization_batchnorm_readvariableop_1_resource:	L
=model_batch_normalization_batchnorm_readvariableop_2_resource:	C
/model_transform4_matmul_readvariableop_resource:
?
0model_transform4_biasadd_readvariableop_resource:	C
/model_transform5_matmul_readvariableop_resource:
?
0model_transform5_biasadd_readvariableop_resource:	L
=model_batch_normalization_1_batchnorm_readvariableop_resource:	P
Amodel_batch_normalization_1_batchnorm_mul_readvariableop_resource:	N
?model_batch_normalization_1_batchnorm_readvariableop_1_resource:	N
?model_batch_normalization_1_batchnorm_readvariableop_2_resource:	H
5model_secondaryclasses_matmul_readvariableop_resource:	D
6model_secondaryclasses_biasadd_readvariableop_resource:D
2actualoutputclasses_matmul_readvariableop_resource:A
3actualoutputclasses_biasadd_readvariableop_resource:
identity¢ExperimentEmb/embedding_lookup¢*actualOutputClasses/BiasAdd/ReadVariableOp¢)actualOutputClasses/MatMul/ReadVariableOp¢2model/batch_normalization/batchnorm/ReadVariableOp¢4model/batch_normalization/batchnorm/ReadVariableOp_1¢4model/batch_normalization/batchnorm/ReadVariableOp_2¢6model/batch_normalization/batchnorm/mul/ReadVariableOp¢4model/batch_normalization_1/batchnorm/ReadVariableOp¢6model/batch_normalization_1/batchnorm/ReadVariableOp_1¢6model/batch_normalization_1/batchnorm/ReadVariableOp_2¢8model/batch_normalization_1/batchnorm/mul/ReadVariableOp¢-model/secondaryClasses/BiasAdd/ReadVariableOp¢,model/secondaryClasses/MatMul/ReadVariableOp¢'model/transform1/BiasAdd/ReadVariableOp¢&model/transform1/MatMul/ReadVariableOp¢'model/transform2/BiasAdd/ReadVariableOp¢&model/transform2/MatMul/ReadVariableOp¢'model/transform3/BiasAdd/ReadVariableOp¢&model/transform3/MatMul/ReadVariableOp¢'model/transform4/BiasAdd/ReadVariableOp¢&model/transform4/MatMul/ReadVariableOp¢'model/transform5/BiasAdd/ReadVariableOp¢&model/transform5/MatMul/ReadVariableOpo
tf.split/ConstConst*
_output_shapes
:*
dtype0*-
value$B""            B      Z
tf.split/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
tf.split/splitSplitVinputstf.split/Const:output:0!tf.split/split/split_dim:output:0*
T0*

Tlen0*
_output_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿB:ÿÿÿÿÿÿÿÿÿ*
	num_splitt
ExperimentEmb/CastCasttf.split/split:output:5*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿö
ExperimentEmb/embedding_lookupResourceGather&experimentemb_embedding_lookup_2027474ExperimentEmb/Cast:y:0*
Tindices0*9
_class/
-+loc:@ExperimentEmb/embedding_lookup/2027474*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0Î
'ExperimentEmb/embedding_lookup/IdentityIdentity'ExperimentEmb/embedding_lookup:output:0*
T0*9
_class/
-+loc:@ExperimentEmb/embedding_lookup/2027474*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)ExperimentEmb/embedding_lookup/Identity_1Identity0ExperimentEmb/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
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
valueB:ù
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
value	B :¯
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:¤
reshape/ReshapeReshape2ExperimentEmb/embedding_lookup/Identity_1:output:0reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
getOnlyPositiveValues/ReluRelureshape/Reshape:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform1/MatMul/ReadVariableOpReadVariableOp/model_transform1_matmul_readvariableop_resource*
_output_shapes
:	B*
dtype0
model/transform1/MatMulMatMultf.split/split:output:4.model/transform1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform1/BiasAdd/ReadVariableOpReadVariableOp0model_transform1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform1/BiasAddBiasAdd!model/transform1/MatMul:product:0/model/transform1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform1/SeluSelu!model/transform1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform2/MatMul/ReadVariableOpReadVariableOp/model_transform2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
model/transform2/MatMulMatMul#model/transform1/Selu:activations:0.model/transform2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform2/BiasAdd/ReadVariableOpReadVariableOp0model_transform2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform2/BiasAddBiasAdd!model/transform2/MatMul:product:0/model/transform2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform2/SeluSelu!model/transform2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform3/MatMul/ReadVariableOpReadVariableOp/model_transform3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
model/transform3/MatMulMatMul#model/transform2/Selu:activations:0.model/transform3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform3/BiasAdd/ReadVariableOpReadVariableOp0model_transform3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform3/BiasAddBiasAdd!model/transform3/MatMul:product:0/model/transform3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform3/SeluSelu!model/transform3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
2model/batch_normalization/batchnorm/ReadVariableOpReadVariableOp;model_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0n
)model/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Æ
'model/batch_normalization/batchnorm/addAddV2:model/batch_normalization/batchnorm/ReadVariableOp:value:02model/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
)model/batch_normalization/batchnorm/RsqrtRsqrt+model/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:³
6model/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp?model_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
'model/batch_normalization/batchnorm/mulMul-model/batch_normalization/batchnorm/Rsqrt:y:0>model/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:µ
)model/batch_normalization/batchnorm/mul_1Mul#model/transform3/Selu:activations:0+model/batch_normalization/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
4model/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0Á
)model/batch_normalization/batchnorm/mul_2Mul<model/batch_normalization/batchnorm/ReadVariableOp_1:value:0+model/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:¯
4model/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp=model_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0Á
'model/batch_normalization/batchnorm/subSub<model/batch_normalization/batchnorm/ReadVariableOp_2:value:0-model/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Á
)model/batch_normalization/batchnorm/add_1AddV2-model/batch_normalization/batchnorm/mul_1:z:0+model/batch_normalization/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout/IdentityIdentity-model/batch_normalization/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform4/MatMul/ReadVariableOpReadVariableOp/model_transform4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0¥
model/transform4/MatMulMatMulmodel/dropout/Identity:output:0.model/transform4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform4/BiasAdd/ReadVariableOpReadVariableOp0model_transform4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform4/BiasAddBiasAdd!model/transform4/MatMul:product:0/model/transform4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform4/SeluSelu!model/transform4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&model/transform5/MatMul/ReadVariableOpReadVariableOp/model_transform5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0©
model/transform5/MatMulMatMul#model/transform4/Selu:activations:0.model/transform5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'model/transform5/BiasAdd/ReadVariableOpReadVariableOp0model_transform5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0ª
model/transform5/BiasAddBiasAdd!model/transform5/MatMul:product:0/model/transform5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
model/transform5/SeluSelu!model/transform5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
4model/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp=model_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype0p
+model/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:Ì
)model/batch_normalization_1/batchnorm/addAddV2<model/batch_normalization_1/batchnorm/ReadVariableOp:value:04model/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:
+model/batch_normalization_1/batchnorm/RsqrtRsqrt-model/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes	
:·
8model/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpAmodel_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype0É
)model/batch_normalization_1/batchnorm/mulMul/model/batch_normalization_1/batchnorm/Rsqrt:y:0@model/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:¹
+model/batch_normalization_1/batchnorm/mul_1Mul#model/transform5/Selu:activations:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
6model/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype0Ç
+model/batch_normalization_1/batchnorm/mul_2Mul>model/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0-model/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes	
:³
6model/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp?model_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype0Ç
)model/batch_normalization_1/batchnorm/subSub>model/batch_normalization_1/batchnorm/ReadVariableOp_2:value:0/model/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ç
+model/batch_normalization_1/batchnorm/add_1AddV2/model/batch_normalization_1/batchnorm/mul_1:z:0-model/batch_normalization_1/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/dropout_1/IdentityIdentity/model/batch_normalization_1/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
,model/secondaryClasses/MatMul/ReadVariableOpReadVariableOp5model_secondaryclasses_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0²
model/secondaryClasses/MatMulMatMul!model/dropout_1/Identity:output:04model/secondaryClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-model/secondaryClasses/BiasAdd/ReadVariableOpReadVariableOp6model_secondaryclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
model/secondaryClasses/BiasAddBiasAdd'model/secondaryClasses/MatMul:product:05model/secondaryClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
model/secondaryClasses/SoftmaxSoftmax'model/secondaryClasses/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dot/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :
dot/ExpandDims
ExpandDims(model/secondaryClasses/Softmax:softmax:0dot/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

dot/MatMulBatchMatMulV2(getOnlyPositiveValues/Relu:activations:0dot/ExpandDims:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿL
	dot/ShapeShapedot/MatMul:output:0*
T0*
_output_shapes
:}
dot/SqueezeSqueezedot/MatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

ÿÿÿÿÿÿÿÿÿ
)actualOutputClasses/MatMul/ReadVariableOpReadVariableOp2actualoutputclasses_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
actualOutputClasses/MatMulMatMuldot/Squeeze:output:01actualOutputClasses/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*actualOutputClasses/BiasAdd/ReadVariableOpReadVariableOp3actualoutputclasses_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0²
actualOutputClasses/BiasAddBiasAdd$actualOutputClasses/MatMul:product:02actualOutputClasses/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
actualOutputClasses/SoftmaxSoftmax$actualOutputClasses/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%actualOutputClasses/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿþ
NoOpNoOp^ExperimentEmb/embedding_lookup+^actualOutputClasses/BiasAdd/ReadVariableOp*^actualOutputClasses/MatMul/ReadVariableOp3^model/batch_normalization/batchnorm/ReadVariableOp5^model/batch_normalization/batchnorm/ReadVariableOp_15^model/batch_normalization/batchnorm/ReadVariableOp_27^model/batch_normalization/batchnorm/mul/ReadVariableOp5^model/batch_normalization_1/batchnorm/ReadVariableOp7^model/batch_normalization_1/batchnorm/ReadVariableOp_17^model/batch_normalization_1/batchnorm/ReadVariableOp_29^model/batch_normalization_1/batchnorm/mul/ReadVariableOp.^model/secondaryClasses/BiasAdd/ReadVariableOp-^model/secondaryClasses/MatMul/ReadVariableOp(^model/transform1/BiasAdd/ReadVariableOp'^model/transform1/MatMul/ReadVariableOp(^model/transform2/BiasAdd/ReadVariableOp'^model/transform2/MatMul/ReadVariableOp(^model/transform3/BiasAdd/ReadVariableOp'^model/transform3/MatMul/ReadVariableOp(^model/transform4/BiasAdd/ReadVariableOp'^model/transform4/MatMul/ReadVariableOp(^model/transform5/BiasAdd/ReadVariableOp'^model/transform5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*T
_input_shapesC
A:ÿÿÿÿÿÿÿÿÿG: : : : : : : : : : : : : : : : : : : : : : : 2@
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
'model/transform1/BiasAdd/ReadVariableOp'model/transform1/BiasAdd/ReadVariableOp2P
&model/transform1/MatMul/ReadVariableOp&model/transform1/MatMul/ReadVariableOp2R
'model/transform2/BiasAdd/ReadVariableOp'model/transform2/BiasAdd/ReadVariableOp2P
&model/transform2/MatMul/ReadVariableOp&model/transform2/MatMul/ReadVariableOp2R
'model/transform3/BiasAdd/ReadVariableOp'model/transform3/BiasAdd/ReadVariableOp2P
&model/transform3/MatMul/ReadVariableOp&model/transform3/MatMul/ReadVariableOp2R
'model/transform4/BiasAdd/ReadVariableOp'model/transform4/BiasAdd/ReadVariableOp2P
&model/transform4/MatMul/ReadVariableOp&model/transform4/MatMul/ReadVariableOp2R
'model/transform5/BiasAdd/ReadVariableOp'model/transform5/BiasAdd/ReadVariableOp2P
&model/transform5/MatMul/ReadVariableOp&model/transform5/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
 
_user_specified_nameinputs
¢
Q
%__inference_dot_layer_call_fn_2028079
inputs_0
inputs_1
identity¸
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dot_layer_call_and_return_conditional_losses_2026850`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
û5
Ö	
B__inference_model_layer_call_and_return_conditional_losses_2026302

inputs%
transform1_2026179:	B!
transform1_2026181:	&
transform2_2026196:
!
transform2_2026198:	&
transform3_2026213:
!
transform3_2026215:	*
batch_normalization_2026218:	*
batch_normalization_2026220:	*
batch_normalization_2026222:	*
batch_normalization_2026224:	&
transform4_2026246:
!
transform4_2026248:	&
transform5_2026263:
!
transform5_2026265:	,
batch_normalization_1_2026268:	,
batch_normalization_1_2026270:	,
batch_normalization_1_2026272:	,
batch_normalization_1_2026274:	+
secondaryclasses_2026296:	&
secondaryclasses_2026298:
identity¢+batch_normalization/StatefulPartitionedCall¢-batch_normalization_1/StatefulPartitionedCall¢(secondaryClasses/StatefulPartitionedCall¢"transform1/StatefulPartitionedCall¢"transform2/StatefulPartitionedCall¢"transform3/StatefulPartitionedCall¢"transform4/StatefulPartitionedCall¢"transform5/StatefulPartitionedCallü
"transform1/StatefulPartitionedCallStatefulPartitionedCallinputstransform1_2026179transform1_2026181*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform1_layer_call_and_return_conditional_losses_2026178¡
"transform2/StatefulPartitionedCallStatefulPartitionedCall+transform1/StatefulPartitionedCall:output:0transform2_2026196transform2_2026198*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform2_layer_call_and_return_conditional_losses_2026195¡
"transform3/StatefulPartitionedCallStatefulPartitionedCall+transform2/StatefulPartitionedCall:output:0transform3_2026213transform3_2026215*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform3_layer_call_and_return_conditional_losses_2026212
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall+transform3/StatefulPartitionedCall:output:0batch_normalization_2026218batch_normalization_2026220batch_normalization_2026222batch_normalization_2026224*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2026020æ
dropout/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_layer_call_and_return_conditional_losses_2026232
"transform4/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0transform4_2026246transform4_2026248*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform4_layer_call_and_return_conditional_losses_2026245¡
"transform5/StatefulPartitionedCallStatefulPartitionedCall+transform4/StatefulPartitionedCall:output:0transform5_2026263transform5_2026265*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_transform5_layer_call_and_return_conditional_losses_2026262
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall+transform5/StatefulPartitionedCall:output:0batch_normalization_1_2026268batch_normalization_1_2026270batch_normalization_1_2026272batch_normalization_1_2026274*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2026102ì
dropout_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_dropout_1_layer_call_and_return_conditional_losses_2026282¯
(secondaryClasses/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0secondaryclasses_2026296secondaryclasses_2026298*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2026295
IdentityIdentity1secondaryClasses/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall)^secondaryClasses/StatefulPartitionedCall#^transform1/StatefulPartitionedCall#^transform2/StatefulPartitionedCall#^transform3/StatefulPartitionedCall#^transform4/StatefulPartitionedCall#^transform5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:ÿÿÿÿÿÿÿÿÿB: : : : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2T
(secondaryClasses/StatefulPartitionedCall(secondaryClasses/StatefulPartitionedCall2H
"transform1/StatefulPartitionedCall"transform1/StatefulPartitionedCall2H
"transform2/StatefulPartitionedCall"transform2/StatefulPartitionedCall2H
"transform3/StatefulPartitionedCall"transform3/StatefulPartitionedCall2H
"transform4/StatefulPartitionedCall"transform4/StatefulPartitionedCall2H
"transform5/StatefulPartitionedCall"transform5/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿB
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¸
serving_default¤
=
vec_date1
serving_default_vec_date:0ÿÿÿÿÿÿÿÿÿGG
actualOutputClasses0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:«

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
__call__
+&call_and_return_all_conditional_losses
_default_save_signature"
_tf_keras_network
"
_tf_keras_input_layer
(
	keras_api"
_tf_keras_layer
·

embeddings
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
 layer_with_weights-2
 layer-3
!layer_with_weights-3
!layer-4
"layer-5
#layer_with_weights-4
#layer-6
$layer_with_weights-5
$layer-7
%layer_with_weights-6
%layer-8
&layer-9
'layer_with_weights-7
'layer-10
(	variables
)trainable_variables
*regularization_losses
+	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_network
§
,	variables
-trainable_variables
.regularization_losses
/	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

0kernel
1bias
2	variables
3trainable_variables
4regularization_losses
5	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ï
6iter

7beta_1

8beta_2
	9decay
:learning_ratemá0mâ1mã;mä<må=mæ>mç?mè@méAmêBmëEmìFmíGmîHmïImðJmñMmòNmóvô0võ1vö;v÷<vø=vù>vú?vû@vüAvýBvþEvÿFvGvHvIvJvMvNv"
	optimizer
Î
0
;1
<2
=3
>4
?5
@6
A7
B8
C9
D10
E11
F12
G13
H14
I15
J16
K17
L18
M19
N20
021
122"
trackable_list_wrapper
®
0
;1
<2
=3
>4
?5
@6
A7
B8
E9
F10
G11
H12
I13
J14
M15
N16
017
118"
trackable_list_wrapper
 "
trackable_list_wrapper
Î
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
"
_generic_user_object
,:*
²2ExperimentEmb/embeddings
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
°
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
Ynon_trainable_variables

Zlayers
[metrics
\layer_regularization_losses
]layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
^non_trainable_variables

_layers
`metrics
alayer_regularization_losses
blayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
"
_tf_keras_input_layer
½

;kernel
<bias
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

=kernel
>bias
g	variables
htrainable_variables
iregularization_losses
j	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
½

?kernel
@bias
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
ì
oaxis
	Agamma
Bbeta
Cmoving_mean
Dmoving_variance
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
§
t	variables
utrainable_variables
vregularization_losses
w	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
½

Ekernel
Fbias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
¡__call__
+¢&call_and_return_all_conditional_losses"
_tf_keras_layer
½

Gkernel
Hbias
|	variables
}trainable_variables
~regularization_losses
	keras_api
£__call__
+¤&call_and_return_all_conditional_losses"
_tf_keras_layer
ñ
	axis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance
	variables
trainable_variables
regularization_losses
	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layer
«
	variables
trainable_variables
regularization_losses
	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"
_tf_keras_layer
Á

Mkernel
Nbias
	variables
trainable_variables
regularization_losses
	keras_api
©__call__
+ª&call_and_return_all_conditional_losses"
_tf_keras_layer
¶
;0
<1
=2
>3
?4
@5
A6
B7
C8
D9
E10
F11
G12
H13
I14
J15
K16
L17
M18
N19"
trackable_list_wrapper

;0
<1
=2
>3
?4
@5
A6
B7
E8
F9
G10
H11
I12
J13
M14
N15"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
(	variables
)trainable_variables
*regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
,	variables
-trainable_variables
.regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
,:*2actualOutputClasses/kernel
&:$2actualOutputClasses/bias
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
$:"	B2transform1/kernel
:2transform1/bias
%:#
2transform2/kernel
:2transform2/bias
%:#
2transform3/kernel
:2transform3/bias
(:&2batch_normalization/gamma
':%2batch_normalization/beta
0:. (2batch_normalization/moving_mean
4:2 (2#batch_normalization/moving_variance
%:#
2transform4/kernel
:2transform4/bias
%:#
2transform5/kernel
:2transform5/bias
*:(2batch_normalization_1/gamma
):'2batch_normalization_1/beta
2:0 (2!batch_normalization_1/moving_mean
6:4 (2%batch_normalization_1/moving_variance
*:(	2secondaryClasses/kernel
#:!2secondaryClasses/bias
<
C0
D1
K2
L3"
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
8
0
1
2"
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
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
non_trainable_variables
 layers
¡metrics
 ¢layer_regularization_losses
£layer_metrics
c	variables
dtrainable_variables
eregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¤non_trainable_variables
¥layers
¦metrics
 §layer_regularization_losses
¨layer_metrics
g	variables
htrainable_variables
iregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
k	variables
ltrainable_variables
mregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
A0
B1
C2
D3"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
®non_trainable_variables
¯layers
°metrics
 ±layer_regularization_losses
²layer_metrics
p	variables
qtrainable_variables
rregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
³non_trainable_variables
´layers
µmetrics
 ¶layer_regularization_losses
·layer_metrics
t	variables
utrainable_variables
vregularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
x	variables
ytrainable_variables
zregularization_losses
¡__call__
+¢&call_and_return_all_conditional_losses
'¢"call_and_return_conditional_losses"
_generic_user_object
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
|	variables
}trainable_variables
~regularization_losses
£__call__
+¤&call_and_return_all_conditional_losses
'¤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ânon_trainable_variables
Ãlayers
Ämetrics
 Ålayer_regularization_losses
Ælayer_metrics
	variables
trainable_variables
regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
	variables
trainable_variables
regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ìnon_trainable_variables
Ílayers
Îmetrics
 Ïlayer_regularization_losses
Ðlayer_metrics
	variables
trainable_variables
regularization_losses
©__call__
+ª&call_and_return_all_conditional_losses
'ª"call_and_return_conditional_losses"
_generic_user_object
<
C0
D1
K2
L3"
trackable_list_wrapper
n
0
1
2
 3
!4
"5
#6
$7
%8
&9
'10"
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

Ñtotal

Òcount
Ó	variables
Ô	keras_api"
_tf_keras_metric

Õtrue_positives
Ötrue_negatives
×false_positives
Øfalse_negatives
Ù	variables
Ú	keras_api"
_tf_keras_metric

Ûtrue_positives
Ütrue_negatives
Ýfalse_positives
Þfalse_negatives
ß	variables
à	keras_api"
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
C0
D1"
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
K0
L1"
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
:  (2total
:  (2count
0
Ñ0
Ò1"
trackable_list_wrapper
.
Ó	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
@
Õ0
Ö1
×2
Ø3"
trackable_list_wrapper
.
Ù	variables"
_generic_user_object
:È (2true_positives
:È (2true_negatives
 :È (2false_positives
 :È (2false_negatives
@
Û0
Ü1
Ý2
Þ3"
trackable_list_wrapper
.
ß	variables"
_generic_user_object
1:/
²2Adam/ExperimentEmb/embeddings/m
1:/2!Adam/actualOutputClasses/kernel/m
+:)2Adam/actualOutputClasses/bias/m
):'	B2Adam/transform1/kernel/m
#:!2Adam/transform1/bias/m
*:(
2Adam/transform2/kernel/m
#:!2Adam/transform2/bias/m
*:(
2Adam/transform3/kernel/m
#:!2Adam/transform3/bias/m
-:+2 Adam/batch_normalization/gamma/m
,:*2Adam/batch_normalization/beta/m
*:(
2Adam/transform4/kernel/m
#:!2Adam/transform4/bias/m
*:(
2Adam/transform5/kernel/m
#:!2Adam/transform5/bias/m
/:-2"Adam/batch_normalization_1/gamma/m
.:,2!Adam/batch_normalization_1/beta/m
/:-	2Adam/secondaryClasses/kernel/m
(:&2Adam/secondaryClasses/bias/m
1:/
²2Adam/ExperimentEmb/embeddings/v
1:/2!Adam/actualOutputClasses/kernel/v
+:)2Adam/actualOutputClasses/bias/v
):'	B2Adam/transform1/kernel/v
#:!2Adam/transform1/bias/v
*:(
2Adam/transform2/kernel/v
#:!2Adam/transform2/bias/v
*:(
2Adam/transform3/kernel/v
#:!2Adam/transform3/bias/v
-:+2 Adam/batch_normalization/gamma/v
,:*2Adam/batch_normalization/beta/v
*:(
2Adam/transform4/kernel/v
#:!2Adam/transform4/bias/v
*:(
2Adam/transform5/kernel/v
#:!2Adam/transform5/bias/v
/:-2"Adam/batch_normalization_1/gamma/v
.:,2!Adam/batch_normalization_1/beta/v
/:-	2Adam/secondaryClasses/kernel/v
(:&2Adam/secondaryClasses/bias/v
ò2ï
)__inference_model_1_layer_call_fn_2026919
)__inference_model_1_layer_call_fn_2027411
)__inference_model_1_layer_call_fn_2027462
)__inference_model_1_layer_call_fn_2027173À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
D__inference_model_1_layer_call_and_return_conditional_losses_2027578
D__inference_model_1_layer_call_and_return_conditional_losses_2027736
D__inference_model_1_layer_call_and_return_conditional_losses_2027237
D__inference_model_1_layer_call_and_return_conditional_losses_2027301À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÎBË
"__inference__wrapped_model_2025996vec_date"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ù2Ö
/__inference_ExperimentEmb_layer_call_fn_2027743¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ô2ñ
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2027753¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ó2Ð
)__inference_reshape_layer_call_fn_2027758¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_reshape_layer_call_and_return_conditional_losses_2027771¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
á2Þ
7__inference_getOnlyPositiveValues_layer_call_fn_2027776¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ü2ù
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2027781¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
'__inference_model_layer_call_fn_2026345
'__inference_model_layer_call_fn_2027826
'__inference_model_layer_call_fn_2027871
'__inference_model_layer_call_fn_2026640À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
B__inference_model_layer_call_and_return_conditional_losses_2027951
B__inference_model_layer_call_and_return_conditional_losses_2028073
B__inference_model_layer_call_and_return_conditional_losses_2026694
B__inference_model_layer_call_and_return_conditional_losses_2026748À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ï2Ì
%__inference_dot_layer_call_fn_2028079¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dot_layer_call_and_return_conditional_losses_2028089¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ß2Ü
5__inference_actualOutputClasses_layer_call_fn_2028098¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ú2÷
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2028109¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÍBÊ
%__inference_signature_wrapper_2027360vec_date"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_transform1_layer_call_fn_2028118¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_transform1_layer_call_and_return_conditional_losses_2028129¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_transform2_layer_call_fn_2028138¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_transform2_layer_call_and_return_conditional_losses_2028149¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_transform3_layer_call_fn_2028158¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_transform3_layer_call_and_return_conditional_losses_2028169¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¨2¥
5__inference_batch_normalization_layer_call_fn_2028182
5__inference_batch_normalization_layer_call_fn_2028195´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2028215
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2028249´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
)__inference_dropout_layer_call_fn_2028254
)__inference_dropout_layer_call_fn_2028259´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_layer_call_and_return_conditional_losses_2028264
D__inference_dropout_layer_call_and_return_conditional_losses_2028276´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ö2Ó
,__inference_transform4_layer_call_fn_2028285¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_transform4_layer_call_and_return_conditional_losses_2028296¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ö2Ó
,__inference_transform5_layer_call_fn_2028305¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ñ2î
G__inference_transform5_layer_call_and_return_conditional_losses_2028316¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¬2©
7__inference_batch_normalization_1_layer_call_fn_2028329
7__inference_batch_normalization_1_layer_call_fn_2028342´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
â2ß
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2028362
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2028396´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
+__inference_dropout_1_layer_call_fn_2028401
+__inference_dropout_1_layer_call_fn_2028406´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ê2Ç
F__inference_dropout_1_layer_call_and_return_conditional_losses_2028411
F__inference_dropout_1_layer_call_and_return_conditional_losses_2028423´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ü2Ù
2__inference_secondaryClasses_layer_call_fn_2028432¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷2ô
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2028443¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ®
J__inference_ExperimentEmb_layer_call_and_return_conditional_losses_2027753`/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "*¢'
 
0ÿÿÿÿÿÿÿÿÿ
 
/__inference_ExperimentEmb_layer_call_fn_2027743S/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¾
"__inference__wrapped_model_2025996;<=>?@DACBEFGHLIKJMN011¢.
'¢$
"
vec_dateÿÿÿÿÿÿÿÿÿG
ª "IªF
D
actualOutputClasses-*
actualOutputClassesÿÿÿÿÿÿÿÿÿ°
P__inference_actualOutputClasses_layer_call_and_return_conditional_losses_2028109\01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
5__inference_actualOutputClasses_layer_call_fn_2028098O01/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿº
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2028362dLIKJ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 º
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2028396dKLIJ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_batch_normalization_1_layer_call_fn_2028329WLIKJ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
7__inference_batch_normalization_1_layer_call_fn_2028342WKLIJ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¸
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2028215dDACB4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¸
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2028249dCDAB4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
5__inference_batch_normalization_layer_call_fn_2028182WDACB4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
5__inference_batch_normalization_layer_call_fn_2028195WCDAB4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿÌ
@__inference_dot_layer_call_and_return_conditional_losses_2028089^¢[
T¢Q
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
%__inference_dot_layer_call_fn_2028079z^¢[
T¢Q
OL
&#
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
F__inference_dropout_1_layer_call_and_return_conditional_losses_2028411^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¨
F__inference_dropout_1_layer_call_and_return_conditional_losses_2028423^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_dropout_1_layer_call_fn_2028401Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ
+__inference_dropout_1_layer_call_fn_2028406Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¦
D__inference_dropout_layer_call_and_return_conditional_losses_2028264^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ¦
D__inference_dropout_layer_call_and_return_conditional_losses_2028276^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ~
)__inference_dropout_layer_call_fn_2028254Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ~
)__inference_dropout_layer_call_fn_2028259Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ¶
R__inference_getOnlyPositiveValues_layer_call_and_return_conditional_losses_2027781`3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
7__inference_getOnlyPositiveValues_layer_call_fn_2027776S3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÃ
D__inference_model_1_layer_call_and_return_conditional_losses_2027237{;<=>?@DACBEFGHLIKJMN019¢6
/¢,
"
vec_dateÿÿÿÿÿÿÿÿÿG
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ã
D__inference_model_1_layer_call_and_return_conditional_losses_2027301{;<=>?@CDABEFGHKLIJMN019¢6
/¢,
"
vec_dateÿÿÿÿÿÿÿÿÿG
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
D__inference_model_1_layer_call_and_return_conditional_losses_2027578y;<=>?@DACBEFGHLIKJMN017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿG
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Á
D__inference_model_1_layer_call_and_return_conditional_losses_2027736y;<=>?@CDABEFGHKLIJMN017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿG
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_model_1_layer_call_fn_2026919n;<=>?@DACBEFGHLIKJMN019¢6
/¢,
"
vec_dateÿÿÿÿÿÿÿÿÿG
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_1_layer_call_fn_2027173n;<=>?@CDABEFGHKLIJMN019¢6
/¢,
"
vec_dateÿÿÿÿÿÿÿÿÿG
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_1_layer_call_fn_2027411l;<=>?@DACBEFGHLIKJMN017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿG
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_model_1_layer_call_fn_2027462l;<=>?@CDABEFGHKLIJMN017¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿG
p

 
ª "ÿÿÿÿÿÿÿÿÿ¿
B__inference_model_layer_call_and_return_conditional_losses_2026694y;<=>?@DACBEFGHLIKJMN:¢7
0¢-
# 
	vec_inputÿÿÿÿÿÿÿÿÿB
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¿
B__inference_model_layer_call_and_return_conditional_losses_2026748y;<=>?@CDABEFGHKLIJMN:¢7
0¢-
# 
	vec_inputÿÿÿÿÿÿÿÿÿB
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
B__inference_model_layer_call_and_return_conditional_losses_2027951v;<=>?@DACBEFGHLIKJMN7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿB
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
B__inference_model_layer_call_and_return_conditional_losses_2028073v;<=>?@CDABEFGHKLIJMN7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿB
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
'__inference_model_layer_call_fn_2026345l;<=>?@DACBEFGHLIKJMN:¢7
0¢-
# 
	vec_inputÿÿÿÿÿÿÿÿÿB
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_model_layer_call_fn_2026640l;<=>?@CDABEFGHKLIJMN:¢7
0¢-
# 
	vec_inputÿÿÿÿÿÿÿÿÿB
p

 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_model_layer_call_fn_2027826i;<=>?@DACBEFGHLIKJMN7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿB
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
'__inference_model_layer_call_fn_2027871i;<=>?@CDABEFGHKLIJMN7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿB
p

 
ª "ÿÿÿÿÿÿÿÿÿ©
D__inference_reshape_layer_call_and_return_conditional_losses_2027771a4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_reshape_layer_call_fn_2027758T4¢1
*¢'
%"
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
M__inference_secondaryClasses_layer_call_and_return_conditional_losses_2028443]MN0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
2__inference_secondaryClasses_layer_call_fn_2028432PMN0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÍ
%__inference_signature_wrapper_2027360£;<=>?@DACBEFGHLIKJMN01=¢:
¢ 
3ª0
.
vec_date"
vec_dateÿÿÿÿÿÿÿÿÿG"IªF
D
actualOutputClasses-*
actualOutputClassesÿÿÿÿÿÿÿÿÿ¨
G__inference_transform1_layer_call_and_return_conditional_losses_2028129];</¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿB
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_transform1_layer_call_fn_2028118P;</¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿB
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_transform2_layer_call_and_return_conditional_losses_2028149^=>0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_transform2_layer_call_fn_2028138Q=>0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_transform3_layer_call_and_return_conditional_losses_2028169^?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_transform3_layer_call_fn_2028158Q?@0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_transform4_layer_call_and_return_conditional_losses_2028296^EF0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_transform4_layer_call_fn_2028285QEF0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
G__inference_transform5_layer_call_and_return_conditional_losses_2028316^GH0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_transform5_layer_call_fn_2028305QGH0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ