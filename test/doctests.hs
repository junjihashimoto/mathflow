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
    "src/MathFlow/PyString.hs"
    ]
