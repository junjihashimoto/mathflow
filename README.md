# mathflow

[![Hackage version](https://img.shields.io/hackage/v/mathflow.svg?style=flat)](https://hackage.haskell.org/package/mathflow)  [![Build Status](https://travis-ci.org/junjihashimoto/mathflow.png?branch=master)](https://travis-ci.org/junjihashimoto/mathflow)

Dependently typed tensorflow modeler

This package provides a model of tensor-operations.

The model's dimensions and the constraints are described by dependent types.

The tensor-operations are based on tensorflow-api.

Currently the model can be translated into python-code.

To write this package, I refer to [this neural network document](https://blog.jle.im/entry/practical-dependent-types-in-haskell-1.html).


# Install

```
> git clone git@github.com:junjihashimoto/mathflow.git
> cd mathflow
> stack install
```

# Usage

```
testMatMul :: Tensor '[2,1] PyString
testMatMul = 
  let n1 = "n1" <-- $(pyConst2 [[2],[3]]) :: Tensor '[2,1] PyString
      n2 = "n2" <-- $(pyConst2 [[2,0],[0,1]]) :: Tensor '[2,2] PyString
      y = "y" <-- (n2 %* n1) :: Tensor '[2,1] PyString
  in y

main = do
  (retcode,stdout,stderr) <- run testMatMul
  print stdout

```
