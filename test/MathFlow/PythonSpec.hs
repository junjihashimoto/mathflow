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


testNet :: Tensor '[1] PyString
testNet = 
  let n1 = "n1" <-- (Tensor "tf.constant(1)") :: Tensor '[1] PyString
      n2 = "n2" <-- (Tensor "tf.constant(2)") :: Tensor '[1] PyString
      n3 = "n3" <-- (Tensor "tf.constant(3)") :: Tensor '[1] PyString
      y = "y" <-- (n1 .+ n2 .+ n3) :: Tensor '[1] PyString
  in y

testMatMul :: Tensor '[2,1] PyString
testMatMul = 
  let n1 = "n1" <-- (Tensor "tf.constant([[2],[3]])") :: Tensor '[2,1] PyString
      n2 = "n2" <-- (Tensor "tf.constant([[1,0],[0,1]])") :: Tensor '[2,2] PyString
      y = "y" <-- (n2 %* n1) :: Tensor '[2,1] PyString
  in y

#ifdef USE_PYTHON
spec = do
  describe "run tensorflow" $ with localhost $ do
    it "command test" $ do
      command "python3" [] (T.unpack src) @>=  exit 0 <> stdout "6\n"
  describe "run pystring" $ with localhost $ do
    it "adder" $ do
      command "python3" [] (toRunnableString (fromTensor testNet)) @>=  exit 0 <> stdout "6\n"
    it "matmul" $ do
      let src = toRunnableString (fromTensor testMatMul)
      liftIO $ putStr src
      command "python3" [] src @>=  exit 0 <> stdout "[[2]\n [3]]\n"
--      runPyString (fromTensor testNet) `shouldReturn` (0,"6\n","")
#else
spec :: Spec
spec = return ()
#endif

