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

module MathFlow where

import GHC.TypeLits
import Data.Proxy
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.TH
import Data.Promotion.Prelude
import Data.String
import qualified Data.List as L
import Data.Monoid (Monoid,(<>))

type family IsZero (n :: Nat) :: Bool where
  IsZero 0 = True
  IsZero _ = False

type family IsSubSamp (f :: [Nat]) (m :: [Nat]) (n :: [Nat]) :: Bool where
  IsSubSamp (1:fs) (m:ms) (n:ns) = IsSubSamp fs ms ns
  IsSubSamp (f:fs) (m:ms) (n:ns) = ((n * f) :== m) :&& (IsSubSamp fs ms ns)
  IsSubSamp '[] '[] '[] = True
  IsSubSamp _ _ _ = False

type family IsMatMul (m :: [Nat]) (o :: [Nat]) (n :: [Nat]) :: Bool where
  IsMatMul m o n =
    Last n :== Last o :&&
    Last m :== Head (Tail (Reverse o)) :&&
    (Tail (Reverse n)) :== (Tail (Reverse m)) :&&
    (Tail (Tail (Reverse n))) :== (Tail (Tail (Reverse o)))


type family IsSameProduct (m :: [Nat]) (n :: [Nat]) :: Bool where
  IsSameProduct (m:mx) (n:nx) = m :== n :&& (Product mx :== Product nx)
  IsSameProduct mx nx = Product mx :== Product nx


data Tensor (n::[Nat]) a =
    T a
  | TAdd (Tensor n a) (Tensor n a)
  | TSub (Tensor n a) (Tensor n a)
  | TMul (Tensor n a) (Tensor n a)
  | TRep (Tensor (Tail n) a)
  | TTr (Tensor (Reverse n) a)
  | forall o m. (SingI o,SingI m,IsMatMul m o n ~ True) => TMatMul (Tensor m a) (Tensor o a)
  | forall m. (SingI m,IsSameProduct m n ~ True) => TReshape (Tensor m a)
  | forall o m.
    (SingI o,SingI m,
     Last n ~ Last o,
     Last m ~ Head (Tail (Reverse o)),
     (Tail (Reverse n)) ~ (Tail (Reverse m))
    ) =>
    TConv2d (Tensor m a) (Tensor o a)
  | forall f m. (SingI f, SingI m,IsSubSamp f m n ~ True) => TMaxPool (Sing f) (Tensor m a)
  | TSoftMax (Tensor n a)
  | TReLu (Tensor n a)
  | TNorm (Tensor n a)
  | forall f m. (SingI f,SingI m,IsSubSamp f m n ~ True) => TSubSamp (Sing f) (Tensor m a)
  | TFunc String (Tensor n a)
  | TLabel String (Tensor n a)

dim :: (SingI n) => Tensor n a -> [Integer]
dim t = dim' $ ty t
  where
    ty :: (SingI n) => Tensor n a -> Sing n
    ty _ = sing

dim' :: Sing (n::[Nat]) -> [Integer]
dim' t = fromSing t


(.+) :: Tensor n a -> Tensor n a -> Tensor n a 
(.+) = TAdd

(.-) :: Tensor n a -> Tensor n a -> Tensor n a 
(.-) = TSub

(.*) :: Tensor n a -> Tensor n a -> Tensor n a 
(.*) = TMul

(%*) :: forall o m n a. (SingI o,SingI m,SingI n,IsMatMul m o n ~ True)
     => Tensor m a -> Tensor o a -> Tensor n a
(%*) a b = TMatMul a b

(<--) :: String -> Tensor n a  -> Tensor n a 
(<--) = TLabel


class FromTensor a where
  fromTensor :: Tensor n a -> a
  toString :: Tensor n a -> String

data PyString =
  PyString {
     variables :: [String]
  ,  expression :: String
  }
  deriving Show

instance Monoid PyString where
  mempty = ""
  mappend (PyString av ae) (PyString bv be) =  PyString (av <> bv) (ae <> be)

instance IsString PyString where
  fromString a = PyString [] a
    

instance FromTensor PyString where
  fromTensor (T a)  = a
  fromTensor (TAdd a b)  = "tf.add( " <> fromTensor a <> ", " <> fromTensor b <> " )"
  fromTensor (TSub a b)  = "tf.add( " <> fromTensor a <> ", tf.negative( " <> fromTensor b <> " ) )"
  fromTensor (TMul a b)  = "tf.multiply( " <> fromTensor a <> ", " <> fromTensor b <> " )"
  fromTensor (TRep a)  = fromTensor a
  fromTensor (TTr a)  = "tf.transpose( " <> fromTensor a <> " )"
  fromTensor (TLabel str a)  = PyString ((str <> " = " <> e):v) str
    where
      (PyString v e) = fromTensor a
  fromTensor (TMatMul a b)  = "tf.nn.matmul( " <> fromTensor a <> ", " <> fromTensor b <> " )"
  fromTensor (TReshape a)  = "tf.reshape( " <> fromTensor a <> ", " <> fromString (show (dim a)) <> " )"
  fromTensor (TConv2d a b)  = "tf.nn.conv2d( " <>
                              fromTensor b <>
                              ", " <>
                              fromTensor a <>
                              ", " <>
                              fromString (show $ map (const 1) (dim a) ) <>
                              ", padding='SAME' )"
  fromTensor (TMaxPool a b)  = "tf.nn.max_pool( " <>
                               fromTensor b <>
                               ", ksize=" <>
                               fromString (show $ dim' a) <>
                               ", strides=" <>
                               fromString (show $ map (const 1) (dim' a) ) <>
                               ", padding='SAME' )"
  fromTensor (TSoftMax a)  = "tf.nn.softmax( " <> fromTensor a <> " )"
  fromTensor (TReLu a)  = "tf.nn.relu( " <> fromTensor a <> " )"
  fromTensor (TNorm a)  = "tf.nn.lrn( " <> fromTensor a <> " )"
  fromTensor (TSubSamp a b) = undefined
  fromTensor (TFunc a b) = fromString a <> "( " <> fromTensor b <> " )"
  toString a = L.intercalate "\n" $ reverse e ++ [v]
    where
      (PyString e v) = fromTensor a
