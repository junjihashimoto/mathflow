# mathflow(Dependently typed tensorflow modeler)

[![Hackage version](https://img.shields.io/hackage/v/mathflow.svg?style=flat)](https://hackage.haskell.org/package/mathflow)  [![Build Status](https://travis-ci.org/junjihashimoto/mathflow.png?branch=master)](https://travis-ci.org/junjihashimoto/mathflow)

This package provides a model of tensor-operations.
The model is independent from tensorflow-binding of python and haskell, though this package generates python-code.
tensor's dimensions and constraints are described by dependent types.
The tensor-operations are based on tensorflow-api.
Currently the model can be translated into python-code.
To write this package, I refer to [this neural network document](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html) and singletons.


# Install

Install tensorflow of python and this package.

```
> sudo apt install python3 python3-pip
> pip3 install -U pip
> pip3 install tensorflow
> git clone git@github.com:junjihashimoto/mathflow.git
> cd mathflow
> stack install
```

# Usage

## About model

Model has a type of ```Tensor (dimensions:[Nat]) value-type output-type```.

* ```dimensions``` are tensor-dimensions.
* ```value-type``` is a value type like Integer or Float of [tensorflow-data-types](https://www.tensorflow.org/programmers_guide/dims_types). 
* ```output-type``` is a type of code which this package generates. PyString-type is used for generating python-code.

This package makes tensorflow-graph from the mode. The model's endpoint is always a tensor-type.

At first write graph by using arithmetic operators like (+,-,*,/), %* (which is matrix multiply) and tensorflow-functions.
Mathflow.{TF,TF.NN,TF.Train} packages define Tensorflow-functions.

A example is below.

```
testMatMul :: Tensor '[2,1] Int PyString
testMatMul = 
  let n1 = (Tensor "tf.constant([[2],[3]])") :: Tensor '[2,1] Int PyString
      n2 = (Tensor "tf.constant([[2,0],[0,1]])") :: Tensor '[2,2] Int PyString
      y = (n2 %* n1) :: Tensor '[2,1] Int PyString
  in y
```


## Create model and run it

Write tensorflow-model.

```
testMatMul :: Tensor '[2,1] Int PyString
testMatMul = 
  let n1 = (Tensor "tf.constant([[2],[3]])") :: Tensor '[2,1] Int PyString
      n2 = (Tensor "tf.constant([[2,0],[0,1]])") :: Tensor '[2,2] Int PyString
      y = n2 %* n1 :: Tensor '[2,1] Int PyString
  in y
```

Run the model. This ```run``` function generates python-code and excecute the code by python.

```
main = do
  (retcode,stdout,stderr) <- run testMatMul
  print stdout

```
