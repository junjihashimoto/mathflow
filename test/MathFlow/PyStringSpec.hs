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
{-# LANGUAGE QuasiQuotes #-}

module MathFlow.PyStringSpec where

import GHC.TypeLits
import Data.Proxy
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.TH
import Data.Promotion.Prelude
import MathFlow

import Test.Hspec

testSingleNet :: Tensor '[100,10] PyString
testSingleNet = 
  let x = "x" <-- (Tensor "x") :: Tensor '[100,784] PyString
      w = "w" <-- (Tensor "w") :: Tensor '[784,10] PyString
      b = "b" <-- (Tensor "b") :: Tensor '[10] PyString
      z = "z" <-- TRep b :: Tensor '[100,10] PyString
      y' = (x %* w) .+ z :: Tensor '[100,10] PyString
      y = "y" <-- TFunc "softmax" y' :: Tensor '[100,10] PyString
  in y

type IMAGE_SIZE = 32
type IMAGE_SIZE_2 = 16
type IMAGE_SIZE_4 = 8
type BATCH_SIZE = 100

--images :: T' [s,IMAGE_SIZE,IMAGE_SIZE,3]

testConvNet0 :: forall s. (SingI s) => Tensor '[s,IMAGE_SIZE,IMAGE_SIZE,3] PyString -> Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] PyString
testConvNet0 x1 = 
  let k1 = TLabel "k1" (Tensor "") :: Tensor '[5,5,3,64] PyString
      b1 = TLabel "b1" (Tensor "") :: Tensor '[64] PyString
      y1' = (TConv2d x1 k1) :: Tensor '[s,IMAGE_SIZE,IMAGE_SIZE,64] PyString
      y1 = TReLu y1' :: Tensor '[s,IMAGE_SIZE,IMAGE_SIZE,64] PyString
      opt = sing :: Sing '[1,2,2,1]
      y2 = TMaxPool opt y1 :: Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] PyString
      y3 = TLabel "y1" (TNorm y2) :: Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] PyString
  in y3

testConvNet1 :: forall s. (SingI s) => Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] PyString -> Tensor '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] PyString
testConvNet1 x1 = 
  let k1 = Tensor "" :: Tensor '[5,5,64,64] PyString
      b1 = Tensor "" :: Tensor '[64] PyString
      y1' = (TConv2d x1 k1) :: Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] PyString
      y1 = TNorm (TReLu y1') :: Tensor '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] PyString
      opt = sing :: Sing '[1,2,2,1]
      y2 = TMaxPool opt y1 :: Tensor '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] PyString
  in y2

testConvNet2 :: forall s. (SingI s) => Tensor '[s,IMAGE_SIZE_4,IMAGE_SIZE_4,64] PyString -> Tensor '[s,384] PyString
testConvNet2 x' = 
  let x = TReshape x' :: Tensor '[s,IMAGE_SIZE_4*IMAGE_SIZE_4*64] PyString
      w = Tensor "" :: Tensor '[IMAGE_SIZE_4*IMAGE_SIZE_4*64,384] PyString
      b = Tensor "" :: Tensor '[384] PyString
      z = TRep b :: Tensor '[s,384] PyString
      y' = (x %* w) .+ z :: Tensor '[s,384] PyString
      y = TReLu y' :: Tensor '[s,384] PyString
  in y

testConvNet3 :: forall s. (SingI s) => Tensor '[s,384] PyString -> Tensor '[s,192] PyString
testConvNet3 x = 
  let w = Tensor "" :: Tensor '[384,192] PyString
      b = Tensor "" :: Tensor '[192] PyString
      z = TRep b :: Tensor '[s,192] PyString
      y' = (x %* w) .+ z :: Tensor '[s,192] PyString
      y = TReLu y' :: Tensor '[s,192] PyString
  in y

testConvNet4 :: forall s. (SingI s) => Tensor '[s,192] PyString -> Tensor '[s,10] PyString
testConvNet4 x = 
  let w = Tensor "" :: Tensor '[192,10] PyString
      b = Tensor "" :: Tensor '[10] PyString
      z = TRep b :: Tensor '[s,10] PyString
      y = (x %* w) .+ z :: Tensor '[s,10] PyString
  in y

testImage :: Tensor '[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3] PyString
testImage = Tensor ""

testConvNet :: Tensor '[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3] PyString -> Tensor '[BATCH_SIZE,10] PyString
testConvNet = testConvNet4.testConvNet3.testConvNet2.testConvNet1.testConvNet0

spec = do
  describe "model to value" $ do
    it "single layer net" $ do
      fromTensor testSingleNet `shouldBe` PyString {variables = ["y = softmax( tf.add( tf.matmul( x, w ), z ) )","x = x","w = w","z = b","b = b"], expression = "y"}
    it "multible layer net" $ do
      fromTensor (testConvNet testImage) `shouldBe` PyString {variables = ["y1 = tf.nn.lrn( tf.nn.max_pool( tf.nn.relu( tf.nn.conv2d( k1, , [1,1,1,1], padding='SAME' ) ), ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME' ) )","k1 = "], expression = "tf.add( tf.matmul( tf.nn.relu( tf.add( tf.matmul( tf.nn.relu( tf.add( tf.matmul( tf.reshape( tf.nn.max_pool( tf.nn.lrn( tf.nn.relu( tf.nn.conv2d( , y1, [1,1,1,1], padding='SAME' ) ) ), ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME' ), [100,8,8,64] ),  ),  ) ),  ),  ) ),  ),  )"}
  describe "model to string" $ do
    it "single layer net" $ do
      toString testSingleNet `shouldBe` "b = b\nz = b\nw = w\nx = x\ny = softmax( tf.add( tf.matmul( x, w ), z ) )\ny"
    it "multible layer net" $ do
      toString (testConvNet testImage) `shouldBe` "k1 = \ny1 = tf.nn.lrn( tf.nn.max_pool( tf.nn.relu( tf.nn.conv2d( k1, , [1,1,1,1], padding='SAME' ) ), ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME' ) )\ntf.add( tf.matmul( tf.nn.relu( tf.add( tf.matmul( tf.nn.relu( tf.add( tf.matmul( tf.reshape( tf.nn.max_pool( tf.nn.lrn( tf.nn.relu( tf.nn.conv2d( , y1, [1,1,1,1], padding='SAME' ) ) ), ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME' ), [100,8,8,64] ),  ),  ) ),  ),  ) ),  ),  )"


testBuild :: Tensor '[1] PyString
testBuild = (Tensor "tf.constant([1])" :: Tensor '[1] PyString)

testBuild2 :: Tensor '[1] PyString
testBuild2 = $(pyConst [1::Integer])

testBuild3 :: Tensor '[1,1,1] PyString
testBuild3 = $(pyConst3 [[[1]]])
