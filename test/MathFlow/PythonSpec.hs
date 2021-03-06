{-# LANGUAGE QuasiQuotes #-}
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
{-# LANGUAGE CPP #-}

module MathFlow.PythonSpec where

import GHC.TypeLits
import Data.Proxy
import Data.Singletons
import Data.Singletons.TypeLits
import Data.Singletons.TH
import Data.Promotion.Prelude
import MathFlow
import MathFlow.TF

import Test.Hspec
import Test.Hspec.Server
import Text.Shakespeare.Text
import qualified Data.Text.Lazy as T
import Data.Monoid
import Control.Monad.IO.Class

src = [lbt|
          |import tensorflow as tf 
          |num1 = tf.constant(1)
          |num2 = tf.constant(2)
          |num3 = tf.constant(3)
          |num1PlusNum2 = tf.add(num1,num2)
          |num1PlusNum2PlusNum3 = tf.add(num1PlusNum2,num3)
          |sess = tf.Session()
          |result = sess.run(num1PlusNum2PlusNum3)
          |print(result)
          |]


testNet :: Tensor '[1] Float PyString
testNet = 
  let n1 = "n1" <-- (Tensor "tf.constant(1)") :: Tensor '[1] Float PyString
      n2 = "n2" <-- (Tensor "tf.constant(2)") :: Tensor '[1] Float PyString
      n3 = "n3" <-- (Tensor "tf.constant(3)") :: Tensor '[1] Float PyString
      y = "y" <-- (n1 + n2 + n3) :: Tensor '[1] Float PyString
  in y

testSub :: Tensor '[1] Float PyString
testSub = 
  let n1 = "n1" <-- (Tensor "tf.constant(100)") :: Tensor '[1] Float PyString
      n2 = "n2" <-- (Tensor "tf.constant(50)") :: Tensor '[1] Float PyString
      n3 = "n3" <-- (Tensor "tf.constant(2)") :: Tensor '[1] Float PyString
      y = "y" <-- (n3 * (n1 - n2))  :: Tensor '[1] Float PyString
  in y

testMatMul :: Tensor '[2,1] Float PyString
testMatMul = 
  let n1 = "n1" <-- $(pyConst2 [[2],[3]]) :: Tensor '[2,1] Float PyString
      n2 = "n2" <-- $(pyConst2 [[2,0],[0,1]]) :: Tensor '[2,2] Float PyString
      y = "y" <-- (n2 %* n1) :: Tensor '[2,1] Float PyString
  in y

testConcat :: Tensor '[2,2] Float PyString
testConcat = 
  let n1 = "n1" <-- (Tensor "tf.constant([[2],[3]])") :: Tensor '[2,1] Float PyString
      n2 = "n2" <-- (Tensor "tf.constant([[2],[3]])") :: Tensor '[2,1] Float PyString
      y = "y" <-- (TConcat n1 n2) :: Tensor '[2,2] Float PyString
  in y

testReplicate :: Tensor '[2,2] Float PyString
testReplicate = 
  let n1 = "n1" <-- (Tensor "tf.constant([[2],[3]])") :: Tensor '[2,1] Float PyString
      n2 = "n2" <-- (Tensor "tf.constant([[2],[3]])") :: Tensor '[2,1] Float PyString
      y = "y" <-- (TConcat n1 n2) :: Tensor '[2,2] Float PyString
  in y

#ifdef USE_PYTHON
spec = do
  describe "run tensorflow" $ with localhost $ do
    it "command test" $ do
      command "python3" [] (T.unpack src) @>=  exit 0 <> stdout "6\n"
  describe "run pystring" $ with localhost $ do
    it "abs" $ do
      let src = toRunnableString (fromTensor (abs' (Tensor "tf.constant(-100)" :: Tensor '[1] Float PyString) "\"x\""))
      liftIO $ putStr src
      command "python3" [] src @>=  exit 0 <> stdout "100\n"
    it "adder" $ do
      command "python3" [] (toRunnableString (fromTensor testNet)) @>=  exit 0 <> stdout "6\n"
    it "subtract" $ do
      command "python3" [] (toRunnableString (fromTensor testSub)) @>=  exit 0 <> stdout "100\n"
    it "matmul" $ do
      let src = toRunnableString (fromTensor testMatMul)
      command "python3" [] src @>=  exit 0 <> stdout "[[4]\n [3]]\n"
    it "concat" $ do
      let src = toRunnableString (fromTensor testConcat)
      liftIO $ putStr src
      command "python3" [] src @>=  exit 0 <> stdout "[[2 2]\n [3 3]]\n"

#else
spec :: Spec
spec = return ()
#endif

