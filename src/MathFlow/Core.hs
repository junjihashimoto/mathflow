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


module MathFlow.Core where

import GHC.TypeLits
import Data.Singletons
import Data.Singletons.TH
import Data.Promotion.Prelude

-- |IsSubSamp // Subsampling constraint
--
-- * (f :: [Nat]) // strides for subsampling
-- * (m :: [Nat]) // dimensions of original tensor 
-- * (n :: [Nat]) // dimensions of subsampled tensor 
-- * :: Bool
type family IsSubSamp (f :: [Nat]) (m :: [Nat]) (n :: [Nat]) :: Bool where
  IsSubSamp (1:fs) (m:ms) (n:ns) = IsSubSamp fs ms ns
  IsSubSamp (f:fs) (m:ms) (n:ns) = ((n * f) :== m) :&& (IsSubSamp fs ms ns)
  IsSubSamp '[] '[] '[] = 'True
  IsSubSamp _ _ _ = 'False

-- |IsMatMul // A constraint for matrix multiplication
--
-- * (m :: [Nat]) // dimensions of a[..., i, k] 
-- * (o :: [Nat]) // dimensions of b[..., k, j]
-- * (n :: [Nat]) // dimensions of output[..., i, j] = sum_k (a[..., i, k] * b[..., k, j]), for all indices i, j.
-- * :: Bool
type family IsMatMul (m :: [Nat]) (o :: [Nat]) (n :: [Nat]) :: Bool where
  IsMatMul m o n =
    Last n :== Last o :&&
    Last m :== Head (Tail (Reverse o)) :&&
    (Tail (Reverse n)) :== (Tail (Reverse m)) :&&
    (Tail (Tail (Reverse n))) :== (Tail (Tail (Reverse o)))

-- |IsConcat // A constraint for concatination of tensor 
--
-- * (m :: [Nat]) // dimensions of a[..., i, ...] 
-- * (o :: [Nat]) // dimensions of b[..., k, ...]
-- * (n :: [Nat]) // dimensions of output[..., i+k, ...] = concat (a,b) 
-- * :: Bool
type family IsConcat (m :: [Nat]) (o :: [Nat]) (n :: [Nat]) :: Bool where
  IsConcat (m:mx) (o:ox) (n:nx) = (m :== o :&& m:== n :|| m + o :== n) :&& IsConcat mx ox nx
  IsConcat '[] '[] '[] = 'True
  IsConcat _ _ _ = 'False

-- |IsSameProduct // A constraint for reshaping tensor
--
-- * (m :: [Nat]) // dimensions of original tensor
-- * (n :: [Nat]) // dimensions of reshaped tensor
-- * :: Bool
type family IsSameProduct (m :: [Nat]) (n :: [Nat]) :: Bool where
  IsSameProduct (m:mx) (n:nx) = m :== n :&& (Product mx :== Product nx)
  IsSameProduct mx nx = Product mx :== Product nx


-- |Dependently typed tensor model
data Tensor (n::[Nat]) a =
    Tensor a -- ^ Transform a value to dependently typed value
  | TAdd (Tensor n a) (Tensor n a)
  | TSub (Tensor n a) (Tensor n a)
  | TMul (Tensor n a) (Tensor n a)
  | TAbs (Tensor n a)
  | TSign (Tensor n a)
  | TRep (Tensor (Tail n) a)
  | TTr (Tensor (Reverse n) a)
  | forall o m. (SingI o,SingI m,SingI n,IsMatMul m o n ~ 'True) => TMatMul (Tensor m a) (Tensor o a)
  | forall o m. (SingI o,SingI m,SingI n,IsConcat m o n ~ 'True) => TConcat (Tensor m a) (Tensor o a)
  | forall m. (SingI m,IsSameProduct m n ~ 'True) => TReshape (Tensor m a)
  | forall o m.
    (SingI o,SingI m,
     Last n ~ Last o,
     Last m ~ Head (Tail (Reverse o)),
     (Tail (Reverse n)) ~ (Tail (Reverse m))
    ) =>
    TConv2d (Tensor m a) (Tensor o a)
  | forall f m. (SingI f, SingI m,IsSubSamp f m n ~ 'True) => TMaxPool (Sing f) (Tensor m a)
  | TSoftMax (Tensor n a)
  | TReLu (Tensor n a)
  | TNorm (Tensor n a)
  | forall f m. (SingI f,SingI m,IsSubSamp f m n ~ 'True) => TSubSamp (Sing f) (Tensor m a)
  | TFunc String (Tensor n a)
  | TLabel String (Tensor n a)


instance Num (Tensor n a) where
  (+) = TAdd
  (-) = TSub
  (*) = TMul
  abs = TAbs
  signum = TSign

-- | get dimension from tensor
-- 
-- >>> dim (Tensor 1 :: Tensor '[192,10] Int)
-- [192,10]
dim :: (SingI n) => Tensor n a -> [Integer]
dim t = dim' $ ty t
  where
    ty :: (SingI n) => Tensor n a -> Sing n
    ty _ = sing

dim' :: Sing (n::[Nat]) -> [Integer]
dim' t = fromSing t

toValue :: forall n a. Sing (n::[Nat]) -> a -> Tensor n a
toValue _ a = Tensor a

--(.+) :: SingI n => Tensor n a -> Tensor n a -> Tensor n a 
--(.+) = TAdd
--
--(.-) :: SingI n => Tensor n a -> Tensor n a -> Tensor n a 
--(.-) = TSub
--
--(.*) :: SingI n => Tensor n a -> Tensor n a -> Tensor n a 
--(.*) = TMul

(%*) :: forall o m n a. (SingI o,SingI m,SingI n,IsMatMul m o n ~ 'True)
     => Tensor m a -> Tensor o a -> Tensor n a
(%*) a b = TMatMul a b

(<--) :: SingI n => String -> Tensor n a  -> Tensor n a 
(<--) = TLabel


class FromTensor a where
  fromTensor :: Tensor n a -> a
  toString :: Tensor n a -> String
  run :: Tensor n a -> IO (Int,String,String)

