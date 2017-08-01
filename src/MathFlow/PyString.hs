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

module MathFlow.PyString where

import Data.Singletons

import Data.String
import qualified Data.List as L
import Data.Monoid (Monoid,(<>))
import MathFlow.Core
import System.Exit
import System.Process

import Language.Haskell.TH

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
  fromTensor (TScalar a) = "tf.constant([" <> fromString (show a) <> "])"
  fromTensor (Tensor a)  = a
  fromTensor v@(TConcat a b)  = wrap v
    where
      wrap :: SingI n => Tensor n t a -> PyString
      wrap t = "tf.concat( [" <> fromTensor a <> ", " <> fromTensor b <> " ]," <> fromString (show (idx (dim t))) <> " )"
      idx i = fst $ head $ filter (\(i,b) -> b ) $ map (\(i,vd,ad) -> (i, vd /= ad)) $ zip3 [0..] i (dim a)
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

  run tensor = do
    (e,stdout,stderr) <- readProcessWithExitCode "python3" [] $ toRunnableString $ fromTensor tensor
    return  (exitCode e,stdout,stderr)
    where
       exitCode e = case e of
         ExitSuccess -> 0
         ExitFailure v -> v

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


-- | Get dimensions of list
--
-- >>> listDim [1]
-- [1]
-- >>> listDim [[1]]
-- [1,1]
-- >>> listDim [[1,2]]
-- [1,2]
-- >>> listDim [[1,2],[1,2]]
-- [2,2]
class ListDimension a where
  listDim :: a -> [Integer]

instance ListDimension Integer where
  listDim _ = []

instance ListDimension a => ListDimension [a] where
  listDim [] = []
  listDim a@(x:xs) = (fromIntegral (length a)) : listDim x

genPyType :: [Integer] -> Type
genPyType dims = (ConT ''Tensor) `AppT` (loop dims) `AppT` (ConT ''Float) `AppT` (ConT ''PyString)
  where
    loop :: [Integer] -> Type
    loop [] = PromotedNilT
    loop (x:xs) = (AppT (AppT PromotedConsT (LitT (NumTyLit x))) (loop xs))

genPyExp :: Show a => a -> Exp
genPyExp values =  (AppE (ConE 'Tensor) (LitE (StringL ("tf.constant(" <> show values <> ")"))))

-- | Gen tensorflow constant expression
--
--  $(pyConst1 [3]) means (Tensor "tf.constant([3])" :: Tensor '[1] PyString)
--  $(pyConst1 [3,3]) means (Tensor "tf.constant([3,3])" :: Tensor '[2] PyString)
--  $(pyConst2 [[3,3],[3,3]]) means (Tensor "tf.constant([[3,3],[3,3]])" :: Tensor '[2,2] PyString)
pyConst1 :: [Integer] -> ExpQ
pyConst1 = pyConst

pyConst2 :: [[Integer]] -> ExpQ
pyConst2 = pyConst

pyConst3 :: [[[Integer]]] -> ExpQ
pyConst3 = pyConst

pyConst4 :: [[[[Integer]]]] -> ExpQ
pyConst4 = pyConst

pyConst :: (Show a,ListDimension a) => a -> ExpQ
pyConst values = return (SigE (genPyExp values) (genPyType (listDim values)))
