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

module MathFlow.CoreSpec where

import GHC.TypeLits
import Data.Proxy
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.TH
import Data.Promotion.Prelude
import MathFlow

import Test.Hspec

testSingleNet :: Tensor '[100,10] Float PyString
testSingleNet = 
  let x = "x" <-- (Tensor "x") :: Tensor '[100,784] Float PyString
      w = "w" <-- (Tensor "w") :: Tensor '[784,10] Float PyString
      b = "b" <-- (Tensor "b") :: Tensor '[10] Float PyString
      z = "z" <-- TRep b :: Tensor '[100,10] Float PyString
      y' = (x %* w) + z :: Tensor '[100,10] Float PyString
      y = "y" <-- TFunc "softmax" y' :: Tensor '[100,10] Float PyString
  in y

type IMAGE_SIZE = 32
type IMAGE_SIZE_2 = 16
type IMAGE_SIZE_4 = 8
type BATCH_SIZE = 100

--images :: T' [s,IMAGE_SIZE,IMAGE_SIZE,3]

testConvNet0 :: forall s. (SingI s) => Tensor '[s,IMAGE_SIZE,IMAGE_SIZE,3] Float PyString -> Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Float PyString
testConvNet0 x1 = 
  let k1 = TLabel "k1" (Tensor "") :: Tensor '[5,5,3,64] Float PyString
      b1 = TLabel "b1" (Tensor "") :: Tensor '[64] Float PyString
      y1' = (TConv2d x1 k1) :: Tensor '[s,IMAGE_SIZE,IMAGE_SIZE,64] Float PyString
      y1 = TReLu y1' :: Tensor '[s,IMAGE_SIZE,IMAGE_SIZE,64] Float PyString
      opt = sing :: Sing '[1,2,2,1]
      y2 = TMaxPool opt y1 :: Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Float PyString
      y3 = TLabel "y1" (TNorm y2) :: Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Float PyString
  in y3

testConvNet1 :: forall s. (SingI s) => Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Float PyString -> Tensor '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] Float PyString
testConvNet1 x1 = 
  let k1 = Tensor "" :: Tensor '[5,5,64,64] Float PyString
      b1 = Tensor "" :: Tensor '[64] Float PyString
      y1' = (TConv2d x1 k1) :: Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Float PyString
      y1 = TNorm (TReLu y1') :: Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Float PyString
      opt = sing :: Sing '[1,2,2,1]
      y2 = TMaxPool opt y1 :: Tensor '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] Float PyString
  in y2

testConvNet2 :: forall s. (SingI s) => Tensor '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] Float PyString -> Tensor '[s,384] Float PyString
testConvNet2 x' = 
  let x = TReshape x' :: Tensor '[s,IMAGE_SIZE_4*IMAGE_SIZE_4*64] Float PyString
      w = Tensor "" :: Tensor '[IMAGE_SIZE_4*IMAGE_SIZE_4*64,384] Float PyString
      b = Tensor "" :: Tensor '[384] Float PyString
      z = TRep b :: Tensor '[s,384] Float PyString
      y' = (x %* w) + z :: Tensor '[s,384] Float PyString
      y = TReLu y' :: Tensor '[s,384] Float PyString
  in y

testConvNet3 :: forall s. (SingI s) => Tensor '[s,384] Float PyString -> Tensor '[s,192] Float PyString
testConvNet3 x = 
  let w = Tensor "" :: Tensor '[384,192] Float PyString
      b = Tensor "" :: Tensor '[192] Float PyString
      z = TRep b :: Tensor '[s,192] Float PyString
      y' = (x %* w) + z :: Tensor '[s,192] Float PyString
      y = TReLu y' :: Tensor '[s,192] Float PyString
  in y

testConvNet4 :: forall s. (SingI s) => Tensor '[s,192] Float PyString -> Tensor '[s,10] Float PyString
testConvNet4 x = 
  let w = Tensor "" :: Tensor '[192,10] Float PyString
      b = Tensor "" :: Tensor '[10] Float PyString
      z = TRep b :: Tensor '[s,10] Float PyString
      y = (x %* w) + z :: Tensor '[s,10] Float PyString
  in y

testImage :: Tensor '[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3] Float PyString
testImage = Tensor ""

testConvNet :: Tensor '[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3] Float PyString -> Tensor '[BATCH_SIZE,10] Float PyString
testConvNet = testConvNet4.testConvNet3.testConvNet2.testConvNet1.testConvNet0

spec = do
  describe "tensor dimention" $ do
    it "type to value" $ do
      dim (Tensor "" :: Tensor '[192,10] Float PyString) `shouldBe` [192,10]

