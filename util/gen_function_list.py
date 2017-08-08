#!/usr/bin/python3

import tensorflow as tf
import inspect
import yaml
import re
from typing import Any


#sigs = i.signature(tf)
def genType(arg:str) -> str :
    if arg == 'a' or \
       arg == 'b' or \
       arg == 'x' or \
       arg == 'y' or \
       arg == 'filter' or \
       arg == 'tensor':
        return 'tensor'
    elif arg == 'name':
        return 'string'
    elif arg == 'shape'or \
         arg == 'strides':
        return 'dimensions'
    elif arg == 'dtype':
        return 'type'
    else:
        return 'string'

def genRetType(n:str,arg:str) -> str :
    return 'tensor'


def getFuncType(package):
    members = inspect.getmembers(package)
    members = filter((lambda m: re.match('^[A-Za-z]',m[0]) ),members)
    members = filter((lambda m: inspect.isfunction(m[1]) ),members)
    ret = {}
    for (name,ptr) in members:
        s = inspect.getfullargspec(ptr)
        v = []
        if s.defaults is not None:
            v = list(s.defaults)
        ret[name]={'args':s.args,'defaults':v, 'types': list(map(genType,s.args)), 'rtype': genRetType(name,s.args)}
    return ret
    #print(list(members))
#    with open(n,'w') as f :
#       f.write(yaml.dump(ret,default_flow_style=False));

#        genDef(name,ret[name])
        
def genSym(prefix:str ,n:str,suffix:str) -> str:
    stat=0
    ret=prefix
    i=0
    if n == "Print" or \
       n == "case" or \
       n == "where":
        return ("tf"+n+suffix)
    else:
        while i<len(n):
            if i==0:
                ret += n[i].lower()
            elif n[i] == '_':
                stat = 1
            elif stat == 1:
                ret += n[i].upper()
                stat = 0
            else:
                ret += n[i]
            i=i+1
        ret += suffix
        return ret

def modName(n:str) -> str:
    s = ""
    s += n[0].lower()
    for i in range(len(n)-1):
        s += n[i+1]
    if n == "type":
        s = "type'"
    elif n == "data":
        s = "data'"
    elif n == "default":
        s = "default'"
    elif n == "_":
        s = "_'"
    return s
    

def isReserved(n:str) -> str:
    reserved=["abs","sin","cos","tan","asin","acos","atan"]
    for i in reserved:
        if i == n:
            return True
    return False
    

def genDef(f,prefix,name,defs):
    for d in ["'",""] :
        sym = genSym(prefix,name,d)
        hasSing = False

        if (len(defs['args']) == len(defs['defaults']) and d == "'") or \
           (0 == len(defs['defaults']) and d == "'") or \
           (isReserved(name) and d == "") :
            print('',file=f)
        else:
            print('%s :: ' % sym,end="",file=f)
            if d == "":
                args = defs['args'][:(len(defs['args'])-len(defs['defaults']))]
            else:
                args = defs['args']
            
    
            for (a,t) in zip(args,defs['types']):
                if t == 'dimensions':
                    hasSing = True
            if hasSing:
                print('SingI n => ',end="",file=f)
            
            
            for (a,t) in zip(args,defs['types']):
                if t == 'tensor':
                    print('Tensor n t a -> ',end="",file=f)
                elif t == 'dimensions':
                    print('Sing n -> ',end="",file=f)
                elif t == 'string':
                    print('String -> ',end="",file=f)
                else:
                    print('String -> ',end="",file=f)
        
            if defs['rtype'] == 'tensor':
                print('Tensor n t a ',file=f)
            elif defs['rtype'] == 'dimensions':
                print('Sing n ',file=f)
            elif defs['rtype'] == 'string':
                print('String ',file=f)
            else:
                print('String ',file=f)
        
            print('%s ' % sym,end="",file=f)
            for a in args:
                print('%s ' % modName(a),end="",file=f)
            print('= ',end="",file=f)
        
            print('TSym "tf.%s" ' % (name),end="",file=f)
            l = len(args)
            i = 0
            for (a,t) in zip(args,defs['types']):
                if t == 'tensor':
                    print('<+> TArgT "%s" %s ' % (a,modName(a)),end="",file=f)
                elif t == 'dimensions':
                    print('<+> TArgSing "%s" %s ' % (a,modName(a)),end="",file=f)
                else:
                    print('<+> TArgS "%s" %s ' % (a,modName(a)),end="",file=f)
                i=i+1
            print('',file=f)

with open('../src/MathFlow/TF.hs',"w") as f:
    header = """
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


module MathFlow.TF where

import GHC.TypeLits
import Data.Singletons
import Data.Singletons.TH
import Data.Promotion.Prelude
import MathFlow.Core
import MathFlow.PyString

"""
    m = getFuncType(tf)
    print(header,file=f)
    for i in m :
        genDef(f,"",i,m[i])
        print('',file=f)
with open('../src/MathFlow/TF/NN.hs',"w") as f:
    header = """
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


module MathFlow.TF.NN where

import GHC.TypeLits
import Data.Singletons
import Data.Singletons.TH
import Data.Promotion.Prelude
import MathFlow.Core
import MathFlow.PyString

"""
    m = getFuncType(tf.nn)
    print(header,file=f)
    for i in m :
        genDef(f,"",i,m[i])
        print('',file=f)
with open('../src/MathFlow/TF/Train.hs',"w") as f:
    header = """
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


module MathFlow.TF.Train where

import GHC.TypeLits
import Data.Singletons
import Data.Singletons.TH
import Data.Promotion.Prelude
import MathFlow.Core
import MathFlow.PyString

"""
    m = getFuncType(tf.train)
    print(header,file=f)
    for i in m :
        genDef(f,"",i,m[i])
        print('',file=f)

