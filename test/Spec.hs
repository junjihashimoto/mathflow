{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE DefaultSignatures #-}
{-# LANGUAGE TypeInType #-}

{-# LANGUAGE OverloadedStrings #-}

import GHC.TypeLits
import Data.Proxy
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.TH
import Data.Promotion.Prelude
import MathFlow

testSingleNet :: T '[100,10] PyString
testSingleNet = 
  let x = TLabel "x" (T "x") :: T '[100,784] PyString
      w = TLabel "w" (T "w") :: T '[784,10] PyString
      b = TLabel "b" (T "b") :: T '[10] PyString
      z = TRep b :: T '[100,10] PyString
      y' = (x %* w) .+ z :: T '[100,10] PyString
      y = TFunc "softmax" y' :: T '[100,10] PyString
  in y

type IMAGE_SIZE = 32
type IMAGE_SIZE_2 = 16
type IMAGE_SIZE_4 = 8
type BATCH_SIZE = 100

--images :: T' [s,IMAGE_SIZE,IMAGE_SIZE,3]

testConvNet0 :: forall s. T '[s,IMAGE_SIZE,IMAGE_SIZE,3] Int -> T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int
testConvNet0 x1 = 
  let k1 = TLabel "k1" (T 1) :: T '[5,5,3,64] Int
      b1 = TLabel "b1" (T 1) :: T '[64] Int
      y1' = (TConv2d x1 k1) :: T '[s,IMAGE_SIZE,IMAGE_SIZE,64] Int
      y1 = TReLu y1' :: T '[s,IMAGE_SIZE,IMAGE_SIZE,64] Int
      opt = sing :: Sing '[1,2,2,1]
      y2 = TMaxPool opt y1 :: T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int
      y3 = TLabel "y1" (TNorm y2) :: T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int
  in y3

testConvNet1 :: forall s. T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int -> T '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] Int
testConvNet1 x1 = 
  let k1 = T 1 :: T '[5,5,64,64] Int
      b1 = T 1 :: T '[64] Int
      y1' = (TConv2d x1 k1) :: T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int
      y1 = TNorm (TReLu y1') :: T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int
      opt = sing :: Sing '[1,2,2,1]
      y2 = TMaxPool opt y1 :: T '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] Int
  in y2

testConvNet2 :: forall s. T '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] Int -> T '[s,384] Int
testConvNet2 x' = 
  let x = TReshape x' :: T '[s,IMAGE_SIZE_4*IMAGE_SIZE_4*64] Int
      w = T 1 :: T '[IMAGE_SIZE_4*IMAGE_SIZE_4*64,384] Int
      b = T 1 :: T '[384] Int
      z = TRep b :: T '[s,384] Int
      y' = (x %* w) .+ z :: T '[s,384] Int
      y = TReLu y' :: T '[s,384] Int
  in y

testConvNet3 :: forall s. T '[s,384] Int -> T '[s,192] Int
testConvNet3 x = 
  let w = T 1 :: T '[384,192] Int
      b = T 1 :: T '[192] Int
      z = TRep b :: T '[s,192] Int
      y' = (x %* w) .+ z :: T '[s,192] Int
      y = TReLu y' :: T '[s,192] Int
  in y

testConvNet4 :: forall s. T '[s,192] Int -> T '[s,10] Int
testConvNet4 x = 
  let w = T 1 :: T '[192,10] Int
      b = T 1 :: T '[10] Int
      z = TRep b :: T '[s,10] Int
      y = (x %* w) .+ z :: T '[s,10] Int
  in y

testConvNet :: T '[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3] Int -> T '[BATCH_SIZE,10] Int
testConvNet = testConvNet4.testConvNet3.testConvNet2.testConvNet1.testConvNet0

main :: IO ()
main = do
  print $ fromTensor testSingleNet
--  print $ fromTensor (testConvNet (T 1))
