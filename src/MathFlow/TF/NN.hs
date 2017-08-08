
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


allCandidateSampler' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
allCandidateSampler' true_classes num_true num_sampled unique seed name = TSym "tf.all_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "seed" seed <+> TArgS "name" name 
allCandidateSampler :: String -> String -> String -> String -> Tensor n t a 
allCandidateSampler true_classes num_true num_sampled unique = TSym "tf.all_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique 

atrousConv2d' :: String -> String -> String -> String -> String -> Tensor n t a 
atrousConv2d' value filters rate padding name = TSym "tf.atrous_conv2d" <+> TArgS "value" value <+> TArgS "filters" filters <+> TArgS "rate" rate <+> TArgS "padding" padding <+> TArgS "name" name 
atrousConv2d :: String -> String -> String -> String -> Tensor n t a 
atrousConv2d value filters rate padding = TSym "tf.atrous_conv2d" <+> TArgS "value" value <+> TArgS "filters" filters <+> TArgS "rate" rate <+> TArgS "padding" padding 

atrousConv2dTranspose' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
atrousConv2dTranspose' value filters output_shape rate padding name = TSym "tf.atrous_conv2d_transpose" <+> TArgS "value" value <+> TArgS "filters" filters <+> TArgS "output_shape" output_shape <+> TArgS "rate" rate <+> TArgS "padding" padding <+> TArgS "name" name 
atrousConv2dTranspose :: String -> String -> String -> String -> String -> Tensor n t a 
atrousConv2dTranspose value filters output_shape rate padding = TSym "tf.atrous_conv2d_transpose" <+> TArgS "value" value <+> TArgS "filters" filters <+> TArgS "output_shape" output_shape <+> TArgS "rate" rate <+> TArgS "padding" padding 

avgPool' :: SingI n => String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
avgPool' value ksize strides padding data_format name = TSym "tf.avg_pool" <+> TArgS "value" value <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
avgPool :: SingI n => String -> String -> Sing n -> String -> Tensor n t a 
avgPool value ksize strides padding = TSym "tf.avg_pool" <+> TArgS "value" value <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding 

avgPool3d' :: SingI n => String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
avgPool3d' input ksize strides padding data_format name = TSym "tf.avg_pool3d" <+> TArgS "input" input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
avgPool3d :: SingI n => String -> String -> Sing n -> String -> Tensor n t a 
avgPool3d input ksize strides padding = TSym "tf.avg_pool3d" <+> TArgS "input" input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding 

batchNormWithGlobalNormalization' :: String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
batchNormWithGlobalNormalization' t m v beta gamma variance_epsilon scale_after_normalization name = TSym "tf.batch_norm_with_global_normalization" <+> TArgS "t" t <+> TArgS "m" m <+> TArgS "v" v <+> TArgS "beta" beta <+> TArgS "gamma" gamma <+> TArgS "variance_epsilon" variance_epsilon <+> TArgS "scale_after_normalization" scale_after_normalization <+> TArgS "name" name 
batchNormWithGlobalNormalization :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
batchNormWithGlobalNormalization t m v beta gamma variance_epsilon scale_after_normalization = TSym "tf.batch_norm_with_global_normalization" <+> TArgS "t" t <+> TArgS "m" m <+> TArgS "v" v <+> TArgS "beta" beta <+> TArgS "gamma" gamma <+> TArgS "variance_epsilon" variance_epsilon <+> TArgS "scale_after_normalization" scale_after_normalization 

batchNormalization' :: Tensor n t a -> String -> String -> String -> String -> String -> String -> Tensor n t a 
batchNormalization' x mean variance offset scale variance_epsilon name = TSym "tf.batch_normalization" <+> TArgT "x" x <+> TArgS "mean" mean <+> TArgS "variance" variance <+> TArgS "offset" offset <+> TArgS "scale" scale <+> TArgS "variance_epsilon" variance_epsilon <+> TArgS "name" name 
batchNormalization :: Tensor n t a -> String -> String -> String -> String -> String -> Tensor n t a 
batchNormalization x mean variance offset scale variance_epsilon = TSym "tf.batch_normalization" <+> TArgT "x" x <+> TArgS "mean" mean <+> TArgS "variance" variance <+> TArgS "offset" offset <+> TArgS "scale" scale <+> TArgS "variance_epsilon" variance_epsilon 

biasAdd' :: String -> String -> String -> String -> Tensor n t a 
biasAdd' value bias data_format name = TSym "tf.bias_add" <+> TArgS "value" value <+> TArgS "bias" bias <+> TArgS "data_format" data_format <+> TArgS "name" name 
biasAdd :: String -> String -> Tensor n t a 
biasAdd value bias = TSym "tf.bias_add" <+> TArgS "value" value <+> TArgS "bias" bias 

bidirectionalDynamicRnn' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
bidirectionalDynamicRnn' cell_fw cell_bw inputs sequence_length initial_state_fw initial_state_bw dtype parallel_iterations swap_memory time_major scope = TSym "tf.bidirectional_dynamic_rnn" <+> TArgS "cell_fw" cell_fw <+> TArgS "cell_bw" cell_bw <+> TArgS "inputs" inputs <+> TArgS "sequence_length" sequence_length <+> TArgS "initial_state_fw" initial_state_fw <+> TArgS "initial_state_bw" initial_state_bw <+> TArgS "dtype" dtype <+> TArgS "parallel_iterations" parallel_iterations <+> TArgS "swap_memory" swap_memory <+> TArgS "time_major" time_major <+> TArgS "scope" scope 
bidirectionalDynamicRnn :: String -> String -> String -> Tensor n t a 
bidirectionalDynamicRnn cell_fw cell_bw inputs = TSym "tf.bidirectional_dynamic_rnn" <+> TArgS "cell_fw" cell_fw <+> TArgS "cell_bw" cell_bw <+> TArgS "inputs" inputs 

computeAccidentalHits' :: String -> String -> String -> String -> String -> Tensor n t a 
computeAccidentalHits' true_classes sampled_candidates num_true seed name = TSym "tf.compute_accidental_hits" <+> TArgS "true_classes" true_classes <+> TArgS "sampled_candidates" sampled_candidates <+> TArgS "num_true" num_true <+> TArgS "seed" seed <+> TArgS "name" name 
computeAccidentalHits :: String -> String -> String -> Tensor n t a 
computeAccidentalHits true_classes sampled_candidates num_true = TSym "tf.compute_accidental_hits" <+> TArgS "true_classes" true_classes <+> TArgS "sampled_candidates" sampled_candidates <+> TArgS "num_true" num_true 

conv1d' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
conv1d' value filters stride padding use_cudnn_on_gpu data_format name = TSym "tf.conv1d" <+> TArgS "value" value <+> TArgS "filters" filters <+> TArgS "stride" stride <+> TArgS "padding" padding <+> TArgS "use_cudnn_on_gpu" use_cudnn_on_gpu <+> TArgS "data_format" data_format <+> TArgS "name" name 
conv1d :: String -> String -> String -> String -> Tensor n t a 
conv1d value filters stride padding = TSym "tf.conv1d" <+> TArgS "value" value <+> TArgS "filters" filters <+> TArgS "stride" stride <+> TArgS "padding" padding 

conv2d' :: SingI n => String -> Tensor n t a -> Sing n -> String -> String -> String -> String -> Tensor n t a 
conv2d' input filter strides padding use_cudnn_on_gpu data_format name = TSym "tf.conv2d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "use_cudnn_on_gpu" use_cudnn_on_gpu <+> TArgS "data_format" data_format <+> TArgS "name" name 
conv2d :: SingI n => String -> Tensor n t a -> Sing n -> String -> Tensor n t a 
conv2d input filter strides padding = TSym "tf.conv2d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "padding" padding 

conv2dBackpropFilter' :: SingI n => String -> String -> String -> Sing n -> String -> String -> String -> String -> Tensor n t a 
conv2dBackpropFilter' input filter_sizes out_backprop strides padding use_cudnn_on_gpu data_format name = TSym "tf.conv2d_backprop_filter" <+> TArgS "input" input <+> TArgS "filter_sizes" filter_sizes <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "use_cudnn_on_gpu" use_cudnn_on_gpu <+> TArgS "data_format" data_format <+> TArgS "name" name 
conv2dBackpropFilter :: SingI n => String -> String -> String -> Sing n -> String -> Tensor n t a 
conv2dBackpropFilter input filter_sizes out_backprop strides padding = TSym "tf.conv2d_backprop_filter" <+> TArgS "input" input <+> TArgS "filter_sizes" filter_sizes <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding 

conv2dBackpropInput' :: SingI n => String -> Tensor n t a -> String -> Sing n -> String -> String -> String -> String -> Tensor n t a 
conv2dBackpropInput' input_sizes filter out_backprop strides padding use_cudnn_on_gpu data_format name = TSym "tf.conv2d_backprop_input" <+> TArgS "input_sizes" input_sizes <+> TArgT "filter" filter <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "use_cudnn_on_gpu" use_cudnn_on_gpu <+> TArgS "data_format" data_format <+> TArgS "name" name 
conv2dBackpropInput :: SingI n => String -> Tensor n t a -> String -> Sing n -> String -> Tensor n t a 
conv2dBackpropInput input_sizes filter out_backprop strides padding = TSym "tf.conv2d_backprop_input" <+> TArgS "input_sizes" input_sizes <+> TArgT "filter" filter <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding 

conv2dTranspose' :: SingI n => String -> Tensor n t a -> String -> Sing n -> String -> String -> String -> Tensor n t a 
conv2dTranspose' value filter output_shape strides padding data_format name = TSym "tf.conv2d_transpose" <+> TArgS "value" value <+> TArgT "filter" filter <+> TArgS "output_shape" output_shape <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
conv2dTranspose :: SingI n => String -> Tensor n t a -> String -> Sing n -> Tensor n t a 
conv2dTranspose value filter output_shape strides = TSym "tf.conv2d_transpose" <+> TArgS "value" value <+> TArgT "filter" filter <+> TArgS "output_shape" output_shape <+> TArgSing "strides" strides 

conv3d' :: SingI n => String -> Tensor n t a -> Sing n -> String -> String -> String -> Tensor n t a 
conv3d' input filter strides padding data_format name = TSym "tf.conv3d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
conv3d :: SingI n => String -> Tensor n t a -> Sing n -> String -> Tensor n t a 
conv3d input filter strides padding = TSym "tf.conv3d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "padding" padding 

conv3dBackpropFilterV2' :: SingI n => String -> String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
conv3dBackpropFilterV2' input filter_sizes out_backprop strides padding data_format name = TSym "tf.conv3d_backprop_filter_v2" <+> TArgS "input" input <+> TArgS "filter_sizes" filter_sizes <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
conv3dBackpropFilterV2 :: SingI n => String -> String -> String -> Sing n -> String -> Tensor n t a 
conv3dBackpropFilterV2 input filter_sizes out_backprop strides padding = TSym "tf.conv3d_backprop_filter_v2" <+> TArgS "input" input <+> TArgS "filter_sizes" filter_sizes <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding 

conv3dTranspose' :: SingI n => String -> Tensor n t a -> String -> Sing n -> String -> String -> String -> Tensor n t a 
conv3dTranspose' value filter output_shape strides padding data_format name = TSym "tf.conv3d_transpose" <+> TArgS "value" value <+> TArgT "filter" filter <+> TArgS "output_shape" output_shape <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
conv3dTranspose :: SingI n => String -> Tensor n t a -> String -> Sing n -> Tensor n t a 
conv3dTranspose value filter output_shape strides = TSym "tf.conv3d_transpose" <+> TArgS "value" value <+> TArgT "filter" filter <+> TArgS "output_shape" output_shape <+> TArgSing "strides" strides 

convolution' :: SingI n => String -> Tensor n t a -> String -> Sing n -> String -> String -> String -> Tensor n t a 
convolution' input filter padding strides dilation_rate name data_format = TSym "tf.convolution" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgS "padding" padding <+> TArgSing "strides" strides <+> TArgS "dilation_rate" dilation_rate <+> TArgS "name" name <+> TArgS "data_format" data_format 
convolution :: String -> Tensor n t a -> String -> Tensor n t a 
convolution input filter padding = TSym "tf.convolution" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgS "padding" padding 

crelu' :: String -> String -> Tensor n t a 
crelu' features name = TSym "tf.crelu" <+> TArgS "features" features <+> TArgS "name" name 
crelu :: String -> Tensor n t a 
crelu features = TSym "tf.crelu" <+> TArgS "features" features 

ctcBeamSearchDecoder' :: String -> String -> String -> String -> String -> Tensor n t a 
ctcBeamSearchDecoder' inputs sequence_length beam_width top_paths merge_repeated = TSym "tf.ctc_beam_search_decoder" <+> TArgS "inputs" inputs <+> TArgS "sequence_length" sequence_length <+> TArgS "beam_width" beam_width <+> TArgS "top_paths" top_paths <+> TArgS "merge_repeated" merge_repeated 
ctcBeamSearchDecoder :: String -> String -> Tensor n t a 
ctcBeamSearchDecoder inputs sequence_length = TSym "tf.ctc_beam_search_decoder" <+> TArgS "inputs" inputs <+> TArgS "sequence_length" sequence_length 

ctcGreedyDecoder' :: String -> String -> String -> Tensor n t a 
ctcGreedyDecoder' inputs sequence_length merge_repeated = TSym "tf.ctc_greedy_decoder" <+> TArgS "inputs" inputs <+> TArgS "sequence_length" sequence_length <+> TArgS "merge_repeated" merge_repeated 
ctcGreedyDecoder :: String -> String -> Tensor n t a 
ctcGreedyDecoder inputs sequence_length = TSym "tf.ctc_greedy_decoder" <+> TArgS "inputs" inputs <+> TArgS "sequence_length" sequence_length 

ctcLoss' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
ctcLoss' labels inputs sequence_length preprocess_collapse_repeated ctc_merge_repeated ignore_longer_outputs_than_inputs time_major = TSym "tf.ctc_loss" <+> TArgS "labels" labels <+> TArgS "inputs" inputs <+> TArgS "sequence_length" sequence_length <+> TArgS "preprocess_collapse_repeated" preprocess_collapse_repeated <+> TArgS "ctc_merge_repeated" ctc_merge_repeated <+> TArgS "ignore_longer_outputs_than_inputs" ignore_longer_outputs_than_inputs <+> TArgS "time_major" time_major 
ctcLoss :: String -> String -> String -> Tensor n t a 
ctcLoss labels inputs sequence_length = TSym "tf.ctc_loss" <+> TArgS "labels" labels <+> TArgS "inputs" inputs <+> TArgS "sequence_length" sequence_length 

depthwiseConv2d' :: SingI n => String -> Tensor n t a -> Sing n -> String -> String -> String -> String -> Tensor n t a 
depthwiseConv2d' input filter strides padding rate name data_format = TSym "tf.depthwise_conv2d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "rate" rate <+> TArgS "name" name <+> TArgS "data_format" data_format 
depthwiseConv2d :: SingI n => String -> Tensor n t a -> Sing n -> String -> Tensor n t a 
depthwiseConv2d input filter strides padding = TSym "tf.depthwise_conv2d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "padding" padding 

depthwiseConv2dNative' :: SingI n => String -> Tensor n t a -> Sing n -> String -> String -> String -> Tensor n t a 
depthwiseConv2dNative' input filter strides padding data_format name = TSym "tf.depthwise_conv2d_native" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
depthwiseConv2dNative :: SingI n => String -> Tensor n t a -> Sing n -> String -> Tensor n t a 
depthwiseConv2dNative input filter strides padding = TSym "tf.depthwise_conv2d_native" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "padding" padding 

depthwiseConv2dNativeBackpropFilter' :: SingI n => String -> String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
depthwiseConv2dNativeBackpropFilter' input filter_sizes out_backprop strides padding data_format name = TSym "tf.depthwise_conv2d_native_backprop_filter" <+> TArgS "input" input <+> TArgS "filter_sizes" filter_sizes <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
depthwiseConv2dNativeBackpropFilter :: SingI n => String -> String -> String -> Sing n -> String -> Tensor n t a 
depthwiseConv2dNativeBackpropFilter input filter_sizes out_backprop strides padding = TSym "tf.depthwise_conv2d_native_backprop_filter" <+> TArgS "input" input <+> TArgS "filter_sizes" filter_sizes <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding 

depthwiseConv2dNativeBackpropInput' :: SingI n => String -> Tensor n t a -> String -> Sing n -> String -> String -> String -> Tensor n t a 
depthwiseConv2dNativeBackpropInput' input_sizes filter out_backprop strides padding data_format name = TSym "tf.depthwise_conv2d_native_backprop_input" <+> TArgS "input_sizes" input_sizes <+> TArgT "filter" filter <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
depthwiseConv2dNativeBackpropInput :: SingI n => String -> Tensor n t a -> String -> Sing n -> String -> Tensor n t a 
depthwiseConv2dNativeBackpropInput input_sizes filter out_backprop strides padding = TSym "tf.depthwise_conv2d_native_backprop_input" <+> TArgS "input_sizes" input_sizes <+> TArgT "filter" filter <+> TArgS "out_backprop" out_backprop <+> TArgSing "strides" strides <+> TArgS "padding" padding 

dilation2d' :: SingI n => String -> Tensor n t a -> Sing n -> String -> String -> String -> Tensor n t a 
dilation2d' input filter strides rates padding name = TSym "tf.dilation2d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "rates" rates <+> TArgS "padding" padding <+> TArgS "name" name 
dilation2d :: SingI n => String -> Tensor n t a -> Sing n -> String -> String -> Tensor n t a 
dilation2d input filter strides rates padding = TSym "tf.dilation2d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgSing "strides" strides <+> TArgS "rates" rates <+> TArgS "padding" padding 

dropout' :: Tensor n t a -> String -> String -> String -> String -> Tensor n t a 
dropout' x keep_prob noise_shape seed name = TSym "tf.dropout" <+> TArgT "x" x <+> TArgS "keep_prob" keep_prob <+> TArgS "noise_shape" noise_shape <+> TArgS "seed" seed <+> TArgS "name" name 
dropout :: Tensor n t a -> String -> Tensor n t a 
dropout x keep_prob = TSym "tf.dropout" <+> TArgT "x" x <+> TArgS "keep_prob" keep_prob 

dynamicRnn' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
dynamicRnn' cell inputs sequence_length initial_state dtype parallel_iterations swap_memory time_major scope = TSym "tf.dynamic_rnn" <+> TArgS "cell" cell <+> TArgS "inputs" inputs <+> TArgS "sequence_length" sequence_length <+> TArgS "initial_state" initial_state <+> TArgS "dtype" dtype <+> TArgS "parallel_iterations" parallel_iterations <+> TArgS "swap_memory" swap_memory <+> TArgS "time_major" time_major <+> TArgS "scope" scope 
dynamicRnn :: String -> String -> Tensor n t a 
dynamicRnn cell inputs = TSym "tf.dynamic_rnn" <+> TArgS "cell" cell <+> TArgS "inputs" inputs 

elu' :: String -> String -> Tensor n t a 
elu' features name = TSym "tf.elu" <+> TArgS "features" features <+> TArgS "name" name 
elu :: String -> Tensor n t a 
elu features = TSym "tf.elu" <+> TArgS "features" features 

embeddingLookup' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
embeddingLookup' params ids partition_strategy name validate_indices max_norm = TSym "tf.embedding_lookup" <+> TArgS "params" params <+> TArgS "ids" ids <+> TArgS "partition_strategy" partition_strategy <+> TArgS "name" name <+> TArgS "validate_indices" validate_indices <+> TArgS "max_norm" max_norm 
embeddingLookup :: String -> String -> Tensor n t a 
embeddingLookup params ids = TSym "tf.embedding_lookup" <+> TArgS "params" params <+> TArgS "ids" ids 

embeddingLookupSparse' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
embeddingLookupSparse' params sp_ids sp_weights partition_strategy name combiner max_norm = TSym "tf.embedding_lookup_sparse" <+> TArgS "params" params <+> TArgS "sp_ids" sp_ids <+> TArgS "sp_weights" sp_weights <+> TArgS "partition_strategy" partition_strategy <+> TArgS "name" name <+> TArgS "combiner" combiner <+> TArgS "max_norm" max_norm 
embeddingLookupSparse :: String -> String -> String -> Tensor n t a 
embeddingLookupSparse params sp_ids sp_weights = TSym "tf.embedding_lookup_sparse" <+> TArgS "params" params <+> TArgS "sp_ids" sp_ids <+> TArgS "sp_weights" sp_weights 

erosion2d' :: SingI n => String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
erosion2d' value kernel strides rates padding name = TSym "tf.erosion2d" <+> TArgS "value" value <+> TArgS "kernel" kernel <+> TArgSing "strides" strides <+> TArgS "rates" rates <+> TArgS "padding" padding <+> TArgS "name" name 
erosion2d :: SingI n => String -> String -> Sing n -> String -> String -> Tensor n t a 
erosion2d value kernel strides rates padding = TSym "tf.erosion2d" <+> TArgS "value" value <+> TArgS "kernel" kernel <+> TArgSing "strides" strides <+> TArgS "rates" rates <+> TArgS "padding" padding 

fixedUnigramCandidateSampler' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
fixedUnigramCandidateSampler' true_classes num_true num_sampled unique range_max vocab_file distortion num_reserved_ids num_shards shard unigrams seed name = TSym "tf.fixed_unigram_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "range_max" range_max <+> TArgS "vocab_file" vocab_file <+> TArgS "distortion" distortion <+> TArgS "num_reserved_ids" num_reserved_ids <+> TArgS "num_shards" num_shards <+> TArgS "shard" shard <+> TArgS "unigrams" unigrams <+> TArgS "seed" seed <+> TArgS "name" name 
fixedUnigramCandidateSampler :: String -> String -> String -> String -> String -> Tensor n t a 
fixedUnigramCandidateSampler true_classes num_true num_sampled unique range_max = TSym "tf.fixed_unigram_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "range_max" range_max 

fractionalAvgPool' :: String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
fractionalAvgPool' value pooling_ratio pseudo_random overlapping deterministic seed seed2 name = TSym "tf.fractional_avg_pool" <+> TArgS "value" value <+> TArgS "pooling_ratio" pooling_ratio <+> TArgS "pseudo_random" pseudo_random <+> TArgS "overlapping" overlapping <+> TArgS "deterministic" deterministic <+> TArgS "seed" seed <+> TArgS "seed2" seed2 <+> TArgS "name" name 
fractionalAvgPool :: String -> String -> Tensor n t a 
fractionalAvgPool value pooling_ratio = TSym "tf.fractional_avg_pool" <+> TArgS "value" value <+> TArgS "pooling_ratio" pooling_ratio 

fractionalMaxPool' :: String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
fractionalMaxPool' value pooling_ratio pseudo_random overlapping deterministic seed seed2 name = TSym "tf.fractional_max_pool" <+> TArgS "value" value <+> TArgS "pooling_ratio" pooling_ratio <+> TArgS "pseudo_random" pseudo_random <+> TArgS "overlapping" overlapping <+> TArgS "deterministic" deterministic <+> TArgS "seed" seed <+> TArgS "seed2" seed2 <+> TArgS "name" name 
fractionalMaxPool :: String -> String -> Tensor n t a 
fractionalMaxPool value pooling_ratio = TSym "tf.fractional_max_pool" <+> TArgS "value" value <+> TArgS "pooling_ratio" pooling_ratio 

fusedBatchNorm' :: Tensor n t a -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
fusedBatchNorm' x scale offset mean variance epsilon data_format is_training name = TSym "tf.fused_batch_norm" <+> TArgT "x" x <+> TArgS "scale" scale <+> TArgS "offset" offset <+> TArgS "mean" mean <+> TArgS "variance" variance <+> TArgS "epsilon" epsilon <+> TArgS "data_format" data_format <+> TArgS "is_training" is_training <+> TArgS "name" name 
fusedBatchNorm :: Tensor n t a -> String -> String -> Tensor n t a 
fusedBatchNorm x scale offset = TSym "tf.fused_batch_norm" <+> TArgT "x" x <+> TArgS "scale" scale <+> TArgS "offset" offset 

inTopK' :: String -> String -> String -> String -> Tensor n t a 
inTopK' predictions targets k name = TSym "tf.in_top_k" <+> TArgS "predictions" predictions <+> TArgS "targets" targets <+> TArgS "k" k <+> TArgS "name" name 
inTopK :: String -> String -> String -> Tensor n t a 
inTopK predictions targets k = TSym "tf.in_top_k" <+> TArgS "predictions" predictions <+> TArgS "targets" targets <+> TArgS "k" k 

l2Loss' :: String -> String -> Tensor n t a 
l2Loss' t name = TSym "tf.l2_loss" <+> TArgS "t" t <+> TArgS "name" name 
l2Loss :: String -> Tensor n t a 
l2Loss t = TSym "tf.l2_loss" <+> TArgS "t" t 

l2Normalize' :: Tensor n t a -> String -> String -> String -> Tensor n t a 
l2Normalize' x dim epsilon name = TSym "tf.l2_normalize" <+> TArgT "x" x <+> TArgS "dim" dim <+> TArgS "epsilon" epsilon <+> TArgS "name" name 
l2Normalize :: Tensor n t a -> String -> Tensor n t a 
l2Normalize x dim = TSym "tf.l2_normalize" <+> TArgT "x" x <+> TArgS "dim" dim 

learnedUnigramCandidateSampler' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
learnedUnigramCandidateSampler' true_classes num_true num_sampled unique range_max seed name = TSym "tf.learned_unigram_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "range_max" range_max <+> TArgS "seed" seed <+> TArgS "name" name 
learnedUnigramCandidateSampler :: String -> String -> String -> String -> String -> Tensor n t a 
learnedUnigramCandidateSampler true_classes num_true num_sampled unique range_max = TSym "tf.learned_unigram_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "range_max" range_max 

localResponseNormalization' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
localResponseNormalization' input depth_radius bias alpha beta name = TSym "tf.local_response_normalization" <+> TArgS "input" input <+> TArgS "depth_radius" depth_radius <+> TArgS "bias" bias <+> TArgS "alpha" alpha <+> TArgS "beta" beta <+> TArgS "name" name 
localResponseNormalization :: String -> Tensor n t a 
localResponseNormalization input = TSym "tf.local_response_normalization" <+> TArgS "input" input 

logPoissonLoss' :: String -> String -> String -> String -> Tensor n t a 
logPoissonLoss' targets log_input compute_full_loss name = TSym "tf.log_poisson_loss" <+> TArgS "targets" targets <+> TArgS "log_input" log_input <+> TArgS "compute_full_loss" compute_full_loss <+> TArgS "name" name 
logPoissonLoss :: String -> String -> Tensor n t a 
logPoissonLoss targets log_input = TSym "tf.log_poisson_loss" <+> TArgS "targets" targets <+> TArgS "log_input" log_input 

logSoftmax' :: String -> String -> String -> Tensor n t a 
logSoftmax' logits dim name = TSym "tf.log_softmax" <+> TArgS "logits" logits <+> TArgS "dim" dim <+> TArgS "name" name 
logSoftmax :: String -> Tensor n t a 
logSoftmax logits = TSym "tf.log_softmax" <+> TArgS "logits" logits 

logUniformCandidateSampler' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
logUniformCandidateSampler' true_classes num_true num_sampled unique range_max seed name = TSym "tf.log_uniform_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "range_max" range_max <+> TArgS "seed" seed <+> TArgS "name" name 
logUniformCandidateSampler :: String -> String -> String -> String -> String -> Tensor n t a 
logUniformCandidateSampler true_classes num_true num_sampled unique range_max = TSym "tf.log_uniform_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "range_max" range_max 

lrn' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
lrn' input depth_radius bias alpha beta name = TSym "tf.lrn" <+> TArgS "input" input <+> TArgS "depth_radius" depth_radius <+> TArgS "bias" bias <+> TArgS "alpha" alpha <+> TArgS "beta" beta <+> TArgS "name" name 
lrn :: String -> Tensor n t a 
lrn input = TSym "tf.lrn" <+> TArgS "input" input 

maxPool' :: SingI n => String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
maxPool' value ksize strides padding data_format name = TSym "tf.max_pool" <+> TArgS "value" value <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
maxPool :: SingI n => String -> String -> Sing n -> String -> Tensor n t a 
maxPool value ksize strides padding = TSym "tf.max_pool" <+> TArgS "value" value <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding 

maxPool3d' :: SingI n => String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
maxPool3d' input ksize strides padding data_format name = TSym "tf.max_pool3d" <+> TArgS "input" input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "data_format" data_format <+> TArgS "name" name 
maxPool3d :: SingI n => String -> String -> Sing n -> String -> Tensor n t a 
maxPool3d input ksize strides padding = TSym "tf.max_pool3d" <+> TArgS "input" input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding 

maxPoolWithArgmax' :: SingI n => String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
maxPoolWithArgmax' input ksize strides padding targmax name = TSym "tf.max_pool_with_argmax" <+> TArgS "input" input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "Targmax" targmax <+> TArgS "name" name 
maxPoolWithArgmax :: SingI n => String -> String -> Sing n -> String -> Tensor n t a 
maxPoolWithArgmax input ksize strides padding = TSym "tf.max_pool_with_argmax" <+> TArgS "input" input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding 

moments' :: Tensor n t a -> String -> String -> String -> String -> Tensor n t a 
moments' x axes shift name keep_dims = TSym "tf.moments" <+> TArgT "x" x <+> TArgS "axes" axes <+> TArgS "shift" shift <+> TArgS "name" name <+> TArgS "keep_dims" keep_dims 
moments :: Tensor n t a -> String -> Tensor n t a 
moments x axes = TSym "tf.moments" <+> TArgT "x" x <+> TArgS "axes" axes 

nceLoss' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
nceLoss' weights biases labels inputs num_sampled num_classes num_true sampled_values remove_accidental_hits partition_strategy name = TSym "tf.nce_loss" <+> TArgS "weights" weights <+> TArgS "biases" biases <+> TArgS "labels" labels <+> TArgS "inputs" inputs <+> TArgS "num_sampled" num_sampled <+> TArgS "num_classes" num_classes <+> TArgS "num_true" num_true <+> TArgS "sampled_values" sampled_values <+> TArgS "remove_accidental_hits" remove_accidental_hits <+> TArgS "partition_strategy" partition_strategy <+> TArgS "name" name 
nceLoss :: String -> String -> String -> String -> String -> String -> Tensor n t a 
nceLoss weights biases labels inputs num_sampled num_classes = TSym "tf.nce_loss" <+> TArgS "weights" weights <+> TArgS "biases" biases <+> TArgS "labels" labels <+> TArgS "inputs" inputs <+> TArgS "num_sampled" num_sampled <+> TArgS "num_classes" num_classes 

normalizeMoments' :: String -> String -> String -> String -> String -> Tensor n t a 
normalizeMoments' counts mean_ss variance_ss shift name = TSym "tf.normalize_moments" <+> TArgS "counts" counts <+> TArgS "mean_ss" mean_ss <+> TArgS "variance_ss" variance_ss <+> TArgS "shift" shift <+> TArgS "name" name 
normalizeMoments :: String -> String -> String -> String -> Tensor n t a 
normalizeMoments counts mean_ss variance_ss shift = TSym "tf.normalize_moments" <+> TArgS "counts" counts <+> TArgS "mean_ss" mean_ss <+> TArgS "variance_ss" variance_ss <+> TArgS "shift" shift 

pool' :: SingI n => String -> String -> String -> String -> String -> Sing n -> String -> String -> Tensor n t a 
pool' input window_shape pooling_type padding dilation_rate strides name data_format = TSym "tf.pool" <+> TArgS "input" input <+> TArgS "window_shape" window_shape <+> TArgS "pooling_type" pooling_type <+> TArgS "padding" padding <+> TArgS "dilation_rate" dilation_rate <+> TArgSing "strides" strides <+> TArgS "name" name <+> TArgS "data_format" data_format 
pool :: String -> String -> String -> String -> Tensor n t a 
pool input window_shape pooling_type padding = TSym "tf.pool" <+> TArgS "input" input <+> TArgS "window_shape" window_shape <+> TArgS "pooling_type" pooling_type <+> TArgS "padding" padding 

quantizedAvgPool' :: SingI n => String -> String -> String -> String -> Sing n -> String -> String -> Tensor n t a 
quantizedAvgPool' input min_input max_input ksize strides padding name = TSym "tf.quantized_avg_pool" <+> TArgS "input" input <+> TArgS "min_input" min_input <+> TArgS "max_input" max_input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "name" name 
quantizedAvgPool :: SingI n => String -> String -> String -> String -> Sing n -> String -> Tensor n t a 
quantizedAvgPool input min_input max_input ksize strides padding = TSym "tf.quantized_avg_pool" <+> TArgS "input" input <+> TArgS "min_input" min_input <+> TArgS "max_input" max_input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding 

quantizedConv2d' :: SingI n => String -> Tensor n t a -> String -> String -> String -> String -> Sing n -> String -> String -> String -> Tensor n t a 
quantizedConv2d' input filter min_input max_input min_filter max_filter strides padding out_type name = TSym "tf.quantized_conv2d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgS "min_input" min_input <+> TArgS "max_input" max_input <+> TArgS "min_filter" min_filter <+> TArgS "max_filter" max_filter <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "out_type" out_type <+> TArgS "name" name 
quantizedConv2d :: SingI n => String -> Tensor n t a -> String -> String -> String -> String -> Sing n -> String -> Tensor n t a 
quantizedConv2d input filter min_input max_input min_filter max_filter strides padding = TSym "tf.quantized_conv2d" <+> TArgS "input" input <+> TArgT "filter" filter <+> TArgS "min_input" min_input <+> TArgS "max_input" max_input <+> TArgS "min_filter" min_filter <+> TArgS "max_filter" max_filter <+> TArgSing "strides" strides <+> TArgS "padding" padding 

quantizedMaxPool' :: SingI n => String -> String -> String -> String -> Sing n -> String -> String -> Tensor n t a 
quantizedMaxPool' input min_input max_input ksize strides padding name = TSym "tf.quantized_max_pool" <+> TArgS "input" input <+> TArgS "min_input" min_input <+> TArgS "max_input" max_input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "name" name 
quantizedMaxPool :: SingI n => String -> String -> String -> String -> Sing n -> String -> Tensor n t a 
quantizedMaxPool input min_input max_input ksize strides padding = TSym "tf.quantized_max_pool" <+> TArgS "input" input <+> TArgS "min_input" min_input <+> TArgS "max_input" max_input <+> TArgS "ksize" ksize <+> TArgSing "strides" strides <+> TArgS "padding" padding 

quantizedReluX' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
quantizedReluX' features max_value min_features max_features out_type name = TSym "tf.quantized_relu_x" <+> TArgS "features" features <+> TArgS "max_value" max_value <+> TArgS "min_features" min_features <+> TArgS "max_features" max_features <+> TArgS "out_type" out_type <+> TArgS "name" name 
quantizedReluX :: String -> String -> String -> String -> Tensor n t a 
quantizedReluX features max_value min_features max_features = TSym "tf.quantized_relu_x" <+> TArgS "features" features <+> TArgS "max_value" max_value <+> TArgS "min_features" min_features <+> TArgS "max_features" max_features 

rawRnn' :: String -> String -> String -> String -> String -> Tensor n t a 
rawRnn' cell loop_fn parallel_iterations swap_memory scope = TSym "tf.raw_rnn" <+> TArgS "cell" cell <+> TArgS "loop_fn" loop_fn <+> TArgS "parallel_iterations" parallel_iterations <+> TArgS "swap_memory" swap_memory <+> TArgS "scope" scope 
rawRnn :: String -> String -> Tensor n t a 
rawRnn cell loop_fn = TSym "tf.raw_rnn" <+> TArgS "cell" cell <+> TArgS "loop_fn" loop_fn 

relu' :: String -> String -> Tensor n t a 
relu' features name = TSym "tf.relu" <+> TArgS "features" features <+> TArgS "name" name 
relu :: String -> Tensor n t a 
relu features = TSym "tf.relu" <+> TArgS "features" features 

relu6' :: String -> String -> Tensor n t a 
relu6' features name = TSym "tf.relu6" <+> TArgS "features" features <+> TArgS "name" name 
relu6 :: String -> Tensor n t a 
relu6 features = TSym "tf.relu6" <+> TArgS "features" features 

reluLayer' :: Tensor n t a -> String -> String -> String -> Tensor n t a 
reluLayer' x weights biases name = TSym "tf.relu_layer" <+> TArgT "x" x <+> TArgS "weights" weights <+> TArgS "biases" biases <+> TArgS "name" name 
reluLayer :: Tensor n t a -> String -> String -> Tensor n t a 
reluLayer x weights biases = TSym "tf.relu_layer" <+> TArgT "x" x <+> TArgS "weights" weights <+> TArgS "biases" biases 

sampledSoftmaxLoss' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
sampledSoftmaxLoss' weights biases labels inputs num_sampled num_classes num_true sampled_values remove_accidental_hits partition_strategy name = TSym "tf.sampled_softmax_loss" <+> TArgS "weights" weights <+> TArgS "biases" biases <+> TArgS "labels" labels <+> TArgS "inputs" inputs <+> TArgS "num_sampled" num_sampled <+> TArgS "num_classes" num_classes <+> TArgS "num_true" num_true <+> TArgS "sampled_values" sampled_values <+> TArgS "remove_accidental_hits" remove_accidental_hits <+> TArgS "partition_strategy" partition_strategy <+> TArgS "name" name 
sampledSoftmaxLoss :: String -> String -> String -> String -> String -> String -> Tensor n t a 
sampledSoftmaxLoss weights biases labels inputs num_sampled num_classes = TSym "tf.sampled_softmax_loss" <+> TArgS "weights" weights <+> TArgS "biases" biases <+> TArgS "labels" labels <+> TArgS "inputs" inputs <+> TArgS "num_sampled" num_sampled <+> TArgS "num_classes" num_classes 

separableConv2d' :: SingI n => String -> String -> String -> Sing n -> String -> String -> String -> String -> Tensor n t a 
separableConv2d' input depthwise_filter pointwise_filter strides padding rate name data_format = TSym "tf.separable_conv2d" <+> TArgS "input" input <+> TArgS "depthwise_filter" depthwise_filter <+> TArgS "pointwise_filter" pointwise_filter <+> TArgSing "strides" strides <+> TArgS "padding" padding <+> TArgS "rate" rate <+> TArgS "name" name <+> TArgS "data_format" data_format 
separableConv2d :: SingI n => String -> String -> String -> Sing n -> String -> Tensor n t a 
separableConv2d input depthwise_filter pointwise_filter strides padding = TSym "tf.separable_conv2d" <+> TArgS "input" input <+> TArgS "depthwise_filter" depthwise_filter <+> TArgS "pointwise_filter" pointwise_filter <+> TArgSing "strides" strides <+> TArgS "padding" padding 

sigmoid' :: Tensor n t a -> String -> Tensor n t a 
sigmoid' x name = TSym "tf.sigmoid" <+> TArgT "x" x <+> TArgS "name" name 
sigmoid :: Tensor n t a -> Tensor n t a 
sigmoid x = TSym "tf.sigmoid" <+> TArgT "x" x 


sigmoidCrossEntropyWithLogits :: Tensor n t a 
sigmoidCrossEntropyWithLogits = TSym "tf.sigmoid_cross_entropy_with_logits" 

softmax' :: String -> String -> String -> Tensor n t a 
softmax' logits dim name = TSym "tf.softmax" <+> TArgS "logits" logits <+> TArgS "dim" dim <+> TArgS "name" name 
softmax :: String -> Tensor n t a 
softmax logits = TSym "tf.softmax" <+> TArgS "logits" logits 


softmaxCrossEntropyWithLogits :: Tensor n t a 
softmaxCrossEntropyWithLogits = TSym "tf.softmax_cross_entropy_with_logits" 

softplus' :: String -> String -> Tensor n t a 
softplus' features name = TSym "tf.softplus" <+> TArgS "features" features <+> TArgS "name" name 
softplus :: String -> Tensor n t a 
softplus features = TSym "tf.softplus" <+> TArgS "features" features 

softsign' :: String -> String -> Tensor n t a 
softsign' features name = TSym "tf.softsign" <+> TArgS "features" features <+> TArgS "name" name 
softsign :: String -> Tensor n t a 
softsign features = TSym "tf.softsign" <+> TArgS "features" features 


sparseSoftmaxCrossEntropyWithLogits :: Tensor n t a 
sparseSoftmaxCrossEntropyWithLogits = TSym "tf.sparse_softmax_cross_entropy_with_logits" 

staticBidirectionalRnn' :: String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
staticBidirectionalRnn' cell_fw cell_bw inputs initial_state_fw initial_state_bw dtype sequence_length scope = TSym "tf.static_bidirectional_rnn" <+> TArgS "cell_fw" cell_fw <+> TArgS "cell_bw" cell_bw <+> TArgS "inputs" inputs <+> TArgS "initial_state_fw" initial_state_fw <+> TArgS "initial_state_bw" initial_state_bw <+> TArgS "dtype" dtype <+> TArgS "sequence_length" sequence_length <+> TArgS "scope" scope 
staticBidirectionalRnn :: String -> String -> String -> Tensor n t a 
staticBidirectionalRnn cell_fw cell_bw inputs = TSym "tf.static_bidirectional_rnn" <+> TArgS "cell_fw" cell_fw <+> TArgS "cell_bw" cell_bw <+> TArgS "inputs" inputs 

staticRnn' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
staticRnn' cell inputs initial_state dtype sequence_length scope = TSym "tf.static_rnn" <+> TArgS "cell" cell <+> TArgS "inputs" inputs <+> TArgS "initial_state" initial_state <+> TArgS "dtype" dtype <+> TArgS "sequence_length" sequence_length <+> TArgS "scope" scope 
staticRnn :: String -> String -> Tensor n t a 
staticRnn cell inputs = TSym "tf.static_rnn" <+> TArgS "cell" cell <+> TArgS "inputs" inputs 

staticStateSavingRnn' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
staticStateSavingRnn' cell inputs state_saver state_name sequence_length scope = TSym "tf.static_state_saving_rnn" <+> TArgS "cell" cell <+> TArgS "inputs" inputs <+> TArgS "state_saver" state_saver <+> TArgS "state_name" state_name <+> TArgS "sequence_length" sequence_length <+> TArgS "scope" scope 
staticStateSavingRnn :: String -> String -> String -> String -> Tensor n t a 
staticStateSavingRnn cell inputs state_saver state_name = TSym "tf.static_state_saving_rnn" <+> TArgS "cell" cell <+> TArgS "inputs" inputs <+> TArgS "state_saver" state_saver <+> TArgS "state_name" state_name 

sufficientStatistics' :: Tensor n t a -> String -> String -> String -> String -> Tensor n t a 
sufficientStatistics' x axes shift keep_dims name = TSym "tf.sufficient_statistics" <+> TArgT "x" x <+> TArgS "axes" axes <+> TArgS "shift" shift <+> TArgS "keep_dims" keep_dims <+> TArgS "name" name 
sufficientStatistics :: Tensor n t a -> String -> Tensor n t a 
sufficientStatistics x axes = TSym "tf.sufficient_statistics" <+> TArgT "x" x <+> TArgS "axes" axes 

tanh' :: Tensor n t a -> String -> Tensor n t a 
tanh' x name = TSym "tf.tanh" <+> TArgT "x" x <+> TArgS "name" name 
tanh :: Tensor n t a -> Tensor n t a 
tanh x = TSym "tf.tanh" <+> TArgT "x" x 

topK' :: String -> String -> String -> String -> Tensor n t a 
topK' input k sorted name = TSym "tf.top_k" <+> TArgS "input" input <+> TArgS "k" k <+> TArgS "sorted" sorted <+> TArgS "name" name 
topK :: String -> Tensor n t a 
topK input = TSym "tf.top_k" <+> TArgS "input" input 

uniformCandidateSampler' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
uniformCandidateSampler' true_classes num_true num_sampled unique range_max seed name = TSym "tf.uniform_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "range_max" range_max <+> TArgS "seed" seed <+> TArgS "name" name 
uniformCandidateSampler :: String -> String -> String -> String -> String -> Tensor n t a 
uniformCandidateSampler true_classes num_true num_sampled unique range_max = TSym "tf.uniform_candidate_sampler" <+> TArgS "true_classes" true_classes <+> TArgS "num_true" num_true <+> TArgS "num_sampled" num_sampled <+> TArgS "unique" unique <+> TArgS "range_max" range_max 

weightedCrossEntropyWithLogits' :: String -> String -> String -> String -> Tensor n t a 
weightedCrossEntropyWithLogits' targets logits pos_weight name = TSym "tf.weighted_cross_entropy_with_logits" <+> TArgS "targets" targets <+> TArgS "logits" logits <+> TArgS "pos_weight" pos_weight <+> TArgS "name" name 
weightedCrossEntropyWithLogits :: String -> String -> String -> Tensor n t a 
weightedCrossEntropyWithLogits targets logits pos_weight = TSym "tf.weighted_cross_entropy_with_logits" <+> TArgS "targets" targets <+> TArgS "logits" logits <+> TArgS "pos_weight" pos_weight 

weightedMoments' :: Tensor n t a -> String -> String -> String -> String -> Tensor n t a 
weightedMoments' x axes frequency_weights name keep_dims = TSym "tf.weighted_moments" <+> TArgT "x" x <+> TArgS "axes" axes <+> TArgS "frequency_weights" frequency_weights <+> TArgS "name" name <+> TArgS "keep_dims" keep_dims 
weightedMoments :: Tensor n t a -> String -> String -> Tensor n t a 
weightedMoments x axes frequency_weights = TSym "tf.weighted_moments" <+> TArgT "x" x <+> TArgS "axes" axes <+> TArgS "frequency_weights" frequency_weights 

withSpaceToBatch' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
withSpaceToBatch' input dilation_rate padding op filter_shape spatial_dims data_format = TSym "tf.with_space_to_batch" <+> TArgS "input" input <+> TArgS "dilation_rate" dilation_rate <+> TArgS "padding" padding <+> TArgS "op" op <+> TArgS "filter_shape" filter_shape <+> TArgS "spatial_dims" spatial_dims <+> TArgS "data_format" data_format 
withSpaceToBatch :: String -> String -> String -> String -> Tensor n t a 
withSpaceToBatch input dilation_rate padding op = TSym "tf.with_space_to_batch" <+> TArgS "input" input <+> TArgS "dilation_rate" dilation_rate <+> TArgS "padding" padding <+> TArgS "op" op 

xwPlusB' :: Tensor n t a -> String -> String -> String -> Tensor n t a 
xwPlusB' x weights biases name = TSym "tf.xw_plus_b" <+> TArgT "x" x <+> TArgS "weights" weights <+> TArgS "biases" biases <+> TArgS "name" name 
xwPlusB :: Tensor n t a -> String -> String -> Tensor n t a 
xwPlusB x weights biases = TSym "tf.xw_plus_b" <+> TArgT "x" x <+> TArgS "weights" weights <+> TArgS "biases" biases 

zeroFraction' :: String -> String -> Tensor n t a 
zeroFraction' value name = TSym "tf.zero_fraction" <+> TArgS "value" value <+> TArgS "name" name 
zeroFraction :: String -> Tensor n t a 
zeroFraction value = TSym "tf.zero_fraction" <+> TArgS "value" value 

