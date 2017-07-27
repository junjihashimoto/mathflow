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

import Data.String
import qualified Data.List as L
import Data.Monoid (Monoid,(<>))
import MathFlow.Core
import System.Exit
import System.Process


data PyString =
  PyString {
     variables :: [String]
  ,  expression :: String
  }
  deriving (Show,Eq,Read)

instance Monoid PyString where
  mempty = ""
  mappend (PyString av ae) (PyString bv be) =  PyString (av <> bv) (ae <> be)

instance IsString PyString where
  fromString a = PyString [] a
    
instance FromTensor PyString where
  fromTensor (Tensor a)  = a
  fromTensor (TConcat a b)  = "tf.concat( " <> fromTensor a <> ", " <> fromTensor b <> " )"
  fromTensor (TAdd a b)  = "tf.add( " <> fromTensor a <> ", " <> fromTensor b <> " )"
  fromTensor (TSub a b)  = "tf.add( " <> fromTensor a <> ", tf.negative( " <> fromTensor b <> " ) )"
  fromTensor (TMul a b)  = "tf.multiply( " <> fromTensor a <> ", " <> fromTensor b <> " )"
  fromTensor (TRep a)  = fromTensor a
  fromTensor (TTr a)  = "tf.transpose( " <> fromTensor a <> " )"
  fromTensor (TLabel str a)  = PyString ((str <> " = " <> e):v) str
    where
      (PyString v e) = fromTensor a
  fromTensor (TMatMul a b)  = "tf.matmul( " <> fromTensor a <> ", " <> fromTensor b <> " )"
  fromTensor (TReshape a)  = "tf.reshape( " <> fromTensor a <> ", " <> fromString (show (dim a)) <> " )"
  fromTensor (TConv2d a b)  = "tf.nn.conv2d( " <>
                              fromTensor b <>
                              ", " <>
                              fromTensor a <>
                              ", " <>
                              fromString (show $ map (const (1::Integer)) (dim a) ) <>
                              ", padding='SAME' )"
  fromTensor (TMaxPool a b)  = "tf.nn.max_pool( " <>
                               fromTensor b <>
                               ", ksize=" <>
                               fromString (show $ dim' a) <>
                               ", strides=" <>
                               fromString (show $ map (const (1::Integer)) (dim' a) ) <>
                               ", padding='SAME' )"
  fromTensor (TSoftMax a)  = "tf.nn.softmax( " <> fromTensor a <> " )"
  fromTensor (TReLu a)  = "tf.nn.relu( " <> fromTensor a <> " )"
  fromTensor (TNorm a)  = "tf.nn.lrn( " <> fromTensor a <> " )"
  fromTensor (TSubSamp a b) = undefined
  fromTensor (TFunc a b) = fromString a <> "( " <> fromTensor b <> " )"
  toString a = L.intercalate "\n" $ reverse e ++ [v]
    where
      (PyString e v) = fromTensor a


toRunnableString :: PyString -> String
toRunnableString (PyString env' value) = code
  where
     code = concat [
         "import tensorflow as tf\n",
         (L.intercalate "\n" $ reverse env' ++ [concat ["__value__ = ", value]]) ,
         "\n",
         "sess = tf.Session()\n",
         "result = sess.run(__value__)\n",
         "print(result)\n"
         ]

runPyString :: PyString -> IO (Int,String,String)
runPyString (PyString env' value) = do
  (e,stdout,stderr) <- readProcessWithExitCode "python3" [] code
  return  (exitCode e,stdout,stderr)
  where
     exitCode code = case code of
       ExitSuccess -> 0
       ExitFailure v -> v
     code = concat [
         "import tensorflow as tf\n",
         (L.intercalate "\n" $ reverse env' ++ [concat ["__value__ = ", value]]) ,
         "\n",
         "sess = tf.Session()\n",
         "result = sess.run(__value__)\n",
         "print(result)\n"
         ]
