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

module MathFlow where

import GHC.TypeLits
import Data.Proxy
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.TH
import Data.Promotion.Prelude

type family IsZero (n :: Nat) :: Bool where
  IsZero 0 = True
  IsZero _ = False

type family IsSubSamp (n :: [Nat]) (m :: [Nat]) (o :: [Nat]) :: Bool where
  IsSubSamp (1:fs) (m:ms) (n:ns) = IsSubSamp fs ms ns
  IsSubSamp (f:fs) (m:ms) (n:ns) = ((n * f) :== m) :&& (IsSubSamp fs ms ns)
  IsSubSamp '[] '[] '[] = True
  IsSubSamp _ _ _ = False



data T (n::[Nat]) a =
    T a
  | TAdd (T n a) (T n a)
  | TSub (T n a) (T n a)
  | TMul (T n a) (T n a)
  | TRep (T (Tail n) a)
  | TTr (T (Reverse n) a)
  | forall o m.
    (Last n ~ Last o,
     Last m ~ Head (Tail (Reverse o)),
     (Tail (Reverse n)) ~ (Tail (Reverse m)),
     (Tail (Tail (Reverse n))) ~ (Tail (Tail (Reverse o)))
    ) =>
    TMatMul (T m a) (T o a)
  | forall m. (Product m ~ Product n) =>  TReshape (T m a)
  | forall o m.
    (Last n ~ Last o,
     Last m ~ Head (Tail (Reverse o)),
     (Tail (Reverse n)) ~ (Tail (Reverse m))
    ) =>
    TConv2d (T m a) (T o a)
  | forall f m. (IsSubSamp f m n ~ True) => TMaxPool (Sing f) (T m a)
  | TSoftMax (T n a)
  | TReLu (T n a)
  | TNorm (T n a)
  | forall f m. (IsSubSamp f m n ~ True) => TSubSamp (Sing f) (T m a)
  | TFunc String (T n a)

(.+) :: T n a -> T n a -> T n a 
(.+) = TAdd

(.-) :: T n a -> T n a -> T n a 
(.-) = TSub

(.*) :: T n a -> T n a -> T n a 
(.*) = TMul

(%*) :: forall o m n a.
        (Last n ~ Last o,
         Last m ~ Head (Tail (Reverse o)),
         (Tail (Reverse n)) ~ (Tail (Reverse m)),
         (Tail (Tail (Reverse n))) ~ (Tail (Tail (Reverse o)))
        ) =>
        T m a -> T o a -> T n a
(%*) a b = TMatMul a b


testSingleNet :: T '[s,10] Int
testSingleNet = 
  let x = T 1 :: T '[s,784] Int
      w = T 1 :: T '[784,10] Int
      b = T 1 :: T '[10] Int
      z = TRep b :: T '[s,10] Int
      y' = (x %* w) .+ z :: T '[s,10] Int
      y = TFunc "softmax" y' :: T '[s,10] Int
  in y

type IMAGE_SIZE = 32
type IMAGE_SIZE_2 = 16
type IMAGE_SIZE_4 = 8
type BATCH_SIZE = 100

--images :: T' [s,IMAGE_SIZE,IMAGE_SIZE,3]

testConvNet0 :: forall s. T '[s,IMAGE_SIZE,IMAGE_SIZE,3] Int -> T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int
testConvNet0 x1 = 
  let k1 = T 1 :: T '[5,5,3,64] Int
      b1 = T 1 :: T '[64] Int
      y1' = (TConv2d x1 k1) :: T '[s,IMAGE_SIZE,IMAGE_SIZE,64] Int
      y1 = TReLu y1' :: T '[s,IMAGE_SIZE,IMAGE_SIZE,64] Int
      opt = sing :: Sing '[1,2,2,1]
      y2 = TMaxPool opt y1 :: T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int
      y3 = TNorm y2 :: T '[s,IMAGE_SIZE_2,IMAGE_SIZE_2,64] Int
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

testConvNet2 :: T '[BATCH_SIZE,IMAGE_SIZE_4,IMAGE_SIZE_4,64] Int -> T '[BATCH_SIZE,384] Int
testConvNet2 x' = 
  let x = TReshape x' :: T '[BATCH_SIZE,IMAGE_SIZE_4*IMAGE_SIZE_4*64] Int
      w = T 1 :: T '[IMAGE_SIZE_4*IMAGE_SIZE_4*64,384] Int
      b = T 1 :: T '[384] Int
      z = TRep b :: T '[BATCH_SIZE,384] Int
      y' = (x %* w) .+ z :: T '[BATCH_SIZE,384] Int
      y = TReLu y' :: T '[BATCH_SIZE,384] Int
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

test = print 123
