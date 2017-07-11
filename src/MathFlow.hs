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

type family IsMaxPool (n :: [Nat]) (m :: [Nat]) (o :: [Nat]) :: Bool where
  IsMaxPool (f:fs) (n:ns) (m:ms) = (n * f :== m) :&& (IsMaxPool fs ns ms)
  IsMaxPool '[] '[] '[] = True
  IsMaxPool _ _ _ = False



data T (n::[Nat]) a =
    T a
  | TAdd (T n a) (T n a)
  | TSub (T n a) (T n a)
  | TMul (T n a) (T n a)
  | TRep (T (Tail n) a)
  | TTr (T (Reverse n) a)
  | forall o m.
    (Last n ~ Last o,
     Last m ~ Last (Tail (Reverse o)),
     (Tail (Reverse n)) ~ (Tail (Reverse m)),
     (Tail (Tail (Reverse n))) ~ (Tail (Tail (Reverse o)))
    ) =>
    TMatMul (T m a) (T o a)
  | forall m. (Product m ~ Product n) =>  TReshape (T m a)
  | forall o m.
    (Last n ~ Last o,
     Last m ~ Last (Tail (Reverse o)),
     (Tail (Reverse n)) ~ (Tail (Reverse m))
    ) =>
    TConv2d (T m a) (T o a)
  | forall f m. (IsMaxPool f m n ~ True) => TMaxPool (Sing f) (T m a)
  | TFunc String (T n a)

(.+) :: T n a -> T n a -> T n a 
(.+) = TAdd

(.-) :: T n a -> T n a -> T n a 
(.-) = TSub

(.*) :: T n a -> T n a -> T n a 
(.*) = TMul

(%*) :: forall o m n a.
        (Last n ~ Last o,
         Last m ~ Last (Tail (Reverse o)),
         (Tail (Reverse n)) ~ (Tail (Reverse m)),
         (Tail (Tail (Reverse n))) ~ (Tail (Tail (Reverse o)))
        ) =>
        T m a -> T o a -> T n a
(%*) a b = TMatMul a b


test1 :: T '[s,10] Int
test1 = 
  let x = T 1 :: T '[s,784] Int
      w = T 1 :: T '[784,10] Int
      b = T 1 :: T '[10] Int
      z = TRep b :: T '[s,10] Int
      y' = (x %* w) .+ z :: T '[s,10] Int
      y = TFunc "softmax" y' :: T '[s,10] Int
  in y

--test2 :: T '[s,10] Int
test2 = 
  let x = T 1 :: T '[s,784] Int
      w = T 1 :: T '[784,10] Int
      b = T 1 :: T '[10] Int
      z = TRep b :: T '[s,10] Int
      y' = (x %* w) .+ z :: T '[s,10] Int
      y = TFunc "softmax" y' :: T '[s,10] Int
  in y

test = print 123
