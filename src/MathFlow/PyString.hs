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

module MathFlow.PyString where

import GHC.TypeLits
import Data.Proxy
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.TH
import Data.Promotion.Prelude
import Data.String
import qualified Data.List as L
import Data.Monoid (Monoid,(<>))
import MathFlow.Core


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
