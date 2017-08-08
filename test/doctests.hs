module Main where

import Test.DocTest

main :: IO ()
main = do
  doctest $
    [
    "-XOverloadedStrings",
    "-XScopedTypeVariables",
    "-XTemplateHaskell",
    "-XTypeFamilies",
    "-XGADTs",
    "-XKindSignatures",
    "-XTypeOperators",
    "-XFlexibleContexts",
    "-XRankNTypes",
    "-XUndecidableInstances",
    "-XFlexibleInstances",
    "-XInstanceSigs",
    "-XDefaultSignatures",
    "-XTypeInType",
    "src/MathFlow/Core.hs",
    "src/MathFlow/PyString.hs",
    "src/MathFlow/TF.hs",
    "src/MathFlow/TF/NN.hs",
    "src/MathFlow/TF/Train.hs"
    ]
