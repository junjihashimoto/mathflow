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

spec = do
  describe "run tensorflow" $ with localhost $ do
    it "command test" $ do
      command "python3" [] (T.unpack src) @>=  exit 0 <> stdout "6\n"

