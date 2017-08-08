
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



monitoredTrainingSession :: Tensor n t a 
monitoredTrainingSession = TSym "tf.MonitoredTrainingSession" 


newCheckpointReader :: String -> Tensor n t a 
newCheckpointReader filepattern = TSym "tf.NewCheckpointReader" <+> TArgS "filepattern" filepattern 

addQueueRunner' :: String -> String -> Tensor n t a 
addQueueRunner' qr collection = TSym "tf.add_queue_runner" <+> TArgS "qr" qr <+> TArgS "collection" collection 
addQueueRunner :: String -> Tensor n t a 
addQueueRunner qr = TSym "tf.add_queue_runner" <+> TArgS "qr" qr 


assertGlobalStep :: String -> Tensor n t a 
assertGlobalStep global_step_tensor = TSym "tf.assert_global_step" <+> TArgS "global_step_tensor" global_step_tensor 

basicTrainLoop' :: String -> String -> String -> String -> String -> Tensor n t a 
basicTrainLoop' supervisor train_step_fn args kwargs master = TSym "tf.basic_train_loop" <+> TArgS "supervisor" supervisor <+> TArgS "train_step_fn" train_step_fn <+> TArgS "args" args <+> TArgS "kwargs" kwargs <+> TArgS "master" master 
basicTrainLoop :: String -> String -> Tensor n t a 
basicTrainLoop supervisor train_step_fn = TSym "tf.basic_train_loop" <+> TArgS "supervisor" supervisor <+> TArgS "train_step_fn" train_step_fn 

batch' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
batch' tensors batch_size num_threads capacity enqueue_many shapes dynamic_pad allow_smaller_final_batch shared_name name = TSym "tf.batch" <+> TArgS "tensors" tensors <+> TArgS "batch_size" batch_size <+> TArgS "num_threads" num_threads <+> TArgS "capacity" capacity <+> TArgS "enqueue_many" enqueue_many <+> TArgS "shapes" shapes <+> TArgS "dynamic_pad" dynamic_pad <+> TArgS "allow_smaller_final_batch" allow_smaller_final_batch <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
batch :: String -> String -> Tensor n t a 
batch tensors batch_size = TSym "tf.batch" <+> TArgS "tensors" tensors <+> TArgS "batch_size" batch_size 

batchJoin' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
batchJoin' tensors_list batch_size capacity enqueue_many shapes dynamic_pad allow_smaller_final_batch shared_name name = TSym "tf.batch_join" <+> TArgS "tensors_list" tensors_list <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "enqueue_many" enqueue_many <+> TArgS "shapes" shapes <+> TArgS "dynamic_pad" dynamic_pad <+> TArgS "allow_smaller_final_batch" allow_smaller_final_batch <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
batchJoin :: String -> String -> Tensor n t a 
batchJoin tensors_list batch_size = TSym "tf.batch_join" <+> TArgS "tensors_list" tensors_list <+> TArgS "batch_size" batch_size 


checkpointExists :: String -> Tensor n t a 
checkpointExists checkpoint_prefix = TSym "tf.checkpoint_exists" <+> TArgS "checkpoint_prefix" checkpoint_prefix 


createGlobalStep :: Tensor n t a 
createGlobalStep = TSym "tf.create_global_step" 


doQuantizeTrainingOnGraphdef :: String -> String -> Tensor n t a 
doQuantizeTrainingOnGraphdef input_graph num_bits = TSym "tf.do_quantize_training_on_graphdef" <+> TArgS "input_graph" input_graph <+> TArgS "num_bits" num_bits 

exponentialDecay' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
exponentialDecay' learning_rate global_step decay_steps decay_rate staircase name = TSym "tf.exponential_decay" <+> TArgS "learning_rate" learning_rate <+> TArgS "global_step" global_step <+> TArgS "decay_steps" decay_steps <+> TArgS "decay_rate" decay_rate <+> TArgS "staircase" staircase <+> TArgS "name" name 
exponentialDecay :: String -> String -> String -> String -> Tensor n t a 
exponentialDecay learning_rate global_step decay_steps decay_rate = TSym "tf.exponential_decay" <+> TArgS "learning_rate" learning_rate <+> TArgS "global_step" global_step <+> TArgS "decay_steps" decay_steps <+> TArgS "decay_rate" decay_rate 


exportMetaGraph :: Tensor n t a 
exportMetaGraph = TSym "tf.export_meta_graph" 

generateCheckpointStateProto' :: String -> String -> String -> Tensor n t a 
generateCheckpointStateProto' save_dir model_checkpoint_path all_model_checkpoint_paths = TSym "tf.generate_checkpoint_state_proto" <+> TArgS "save_dir" save_dir <+> TArgS "model_checkpoint_path" model_checkpoint_path <+> TArgS "all_model_checkpoint_paths" all_model_checkpoint_paths 
generateCheckpointStateProto :: String -> String -> Tensor n t a 
generateCheckpointStateProto save_dir model_checkpoint_path = TSym "tf.generate_checkpoint_state_proto" <+> TArgS "save_dir" save_dir <+> TArgS "model_checkpoint_path" model_checkpoint_path 


getCheckpointMtimes :: String -> Tensor n t a 
getCheckpointMtimes checkpoint_prefixes = TSym "tf.get_checkpoint_mtimes" <+> TArgS "checkpoint_prefixes" checkpoint_prefixes 

getCheckpointState' :: String -> String -> Tensor n t a 
getCheckpointState' checkpoint_dir latest_filename = TSym "tf.get_checkpoint_state" <+> TArgS "checkpoint_dir" checkpoint_dir <+> TArgS "latest_filename" latest_filename 
getCheckpointState :: String -> Tensor n t a 
getCheckpointState checkpoint_dir = TSym "tf.get_checkpoint_state" <+> TArgS "checkpoint_dir" checkpoint_dir 


getGlobalStep :: Tensor n t a 
getGlobalStep = TSym "tf.get_global_step" 


getOrCreateGlobalStep :: Tensor n t a 
getOrCreateGlobalStep = TSym "tf.get_or_create_global_step" 


globalStep :: String -> String -> Tensor n t a 
globalStep sess global_step_tensor = TSym "tf.global_step" <+> TArgS "sess" sess <+> TArgS "global_step_tensor" global_step_tensor 

importMetaGraph' :: String -> String -> String -> Tensor n t a 
importMetaGraph' meta_graph_or_file clear_devices import_scope = TSym "tf.import_meta_graph" <+> TArgS "meta_graph_or_file" meta_graph_or_file <+> TArgS "clear_devices" clear_devices <+> TArgS "import_scope" import_scope 
importMetaGraph :: String -> Tensor n t a 
importMetaGraph meta_graph_or_file = TSym "tf.import_meta_graph" <+> TArgS "meta_graph_or_file" meta_graph_or_file 

inputProducer' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
inputProducer' input_tensor element_shape num_epochs shuffle seed capacity shared_name summary_name name cancel_op = TSym "tf.input_producer" <+> TArgS "input_tensor" input_tensor <+> TArgS "element_shape" element_shape <+> TArgS "num_epochs" num_epochs <+> TArgS "shuffle" shuffle <+> TArgS "seed" seed <+> TArgS "capacity" capacity <+> TArgS "shared_name" shared_name <+> TArgS "summary_name" summary_name <+> TArgS "name" name <+> TArgS "cancel_op" cancel_op 
inputProducer :: String -> Tensor n t a 
inputProducer input_tensor = TSym "tf.input_producer" <+> TArgS "input_tensor" input_tensor 

inverseTimeDecay' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
inverseTimeDecay' learning_rate global_step decay_steps decay_rate staircase name = TSym "tf.inverse_time_decay" <+> TArgS "learning_rate" learning_rate <+> TArgS "global_step" global_step <+> TArgS "decay_steps" decay_steps <+> TArgS "decay_rate" decay_rate <+> TArgS "staircase" staircase <+> TArgS "name" name 
inverseTimeDecay :: String -> String -> String -> String -> Tensor n t a 
inverseTimeDecay learning_rate global_step decay_steps decay_rate = TSym "tf.inverse_time_decay" <+> TArgS "learning_rate" learning_rate <+> TArgS "global_step" global_step <+> TArgS "decay_steps" decay_steps <+> TArgS "decay_rate" decay_rate 

latestCheckpoint' :: String -> String -> Tensor n t a 
latestCheckpoint' checkpoint_dir latest_filename = TSym "tf.latest_checkpoint" <+> TArgS "checkpoint_dir" checkpoint_dir <+> TArgS "latest_filename" latest_filename 
latestCheckpoint :: String -> Tensor n t a 
latestCheckpoint checkpoint_dir = TSym "tf.latest_checkpoint" <+> TArgS "checkpoint_dir" checkpoint_dir 

limitEpochs' :: Tensor n t a -> String -> String -> Tensor n t a 
limitEpochs' tensor num_epochs name = TSym "tf.limit_epochs" <+> TArgT "tensor" tensor <+> TArgS "num_epochs" num_epochs <+> TArgS "name" name 
limitEpochs :: Tensor n t a -> Tensor n t a 
limitEpochs tensor = TSym "tf.limit_epochs" <+> TArgT "tensor" tensor 

matchFilenamesOnce' :: String -> String -> Tensor n t a 
matchFilenamesOnce' pattern name = TSym "tf.match_filenames_once" <+> TArgS "pattern" pattern <+> TArgS "name" name 
matchFilenamesOnce :: String -> Tensor n t a 
matchFilenamesOnce pattern = TSym "tf.match_filenames_once" <+> TArgS "pattern" pattern 

maybeBatch' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
maybeBatch' tensors keep_input batch_size num_threads capacity enqueue_many shapes dynamic_pad allow_smaller_final_batch shared_name name = TSym "tf.maybe_batch" <+> TArgS "tensors" tensors <+> TArgS "keep_input" keep_input <+> TArgS "batch_size" batch_size <+> TArgS "num_threads" num_threads <+> TArgS "capacity" capacity <+> TArgS "enqueue_many" enqueue_many <+> TArgS "shapes" shapes <+> TArgS "dynamic_pad" dynamic_pad <+> TArgS "allow_smaller_final_batch" allow_smaller_final_batch <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
maybeBatch :: String -> String -> String -> Tensor n t a 
maybeBatch tensors keep_input batch_size = TSym "tf.maybe_batch" <+> TArgS "tensors" tensors <+> TArgS "keep_input" keep_input <+> TArgS "batch_size" batch_size 

maybeBatchJoin' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
maybeBatchJoin' tensors_list keep_input batch_size capacity enqueue_many shapes dynamic_pad allow_smaller_final_batch shared_name name = TSym "tf.maybe_batch_join" <+> TArgS "tensors_list" tensors_list <+> TArgS "keep_input" keep_input <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "enqueue_many" enqueue_many <+> TArgS "shapes" shapes <+> TArgS "dynamic_pad" dynamic_pad <+> TArgS "allow_smaller_final_batch" allow_smaller_final_batch <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
maybeBatchJoin :: String -> String -> String -> Tensor n t a 
maybeBatchJoin tensors_list keep_input batch_size = TSym "tf.maybe_batch_join" <+> TArgS "tensors_list" tensors_list <+> TArgS "keep_input" keep_input <+> TArgS "batch_size" batch_size 

maybeShuffleBatch' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
maybeShuffleBatch' tensors batch_size capacity min_after_dequeue keep_input num_threads seed enqueue_many shapes allow_smaller_final_batch shared_name name = TSym "tf.maybe_shuffle_batch" <+> TArgS "tensors" tensors <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "min_after_dequeue" min_after_dequeue <+> TArgS "keep_input" keep_input <+> TArgS "num_threads" num_threads <+> TArgS "seed" seed <+> TArgS "enqueue_many" enqueue_many <+> TArgS "shapes" shapes <+> TArgS "allow_smaller_final_batch" allow_smaller_final_batch <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
maybeShuffleBatch :: String -> String -> String -> String -> String -> Tensor n t a 
maybeShuffleBatch tensors batch_size capacity min_after_dequeue keep_input = TSym "tf.maybe_shuffle_batch" <+> TArgS "tensors" tensors <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "min_after_dequeue" min_after_dequeue <+> TArgS "keep_input" keep_input 

maybeShuffleBatchJoin' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
maybeShuffleBatchJoin' tensors_list batch_size capacity min_after_dequeue keep_input seed enqueue_many shapes allow_smaller_final_batch shared_name name = TSym "tf.maybe_shuffle_batch_join" <+> TArgS "tensors_list" tensors_list <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "min_after_dequeue" min_after_dequeue <+> TArgS "keep_input" keep_input <+> TArgS "seed" seed <+> TArgS "enqueue_many" enqueue_many <+> TArgS "shapes" shapes <+> TArgS "allow_smaller_final_batch" allow_smaller_final_batch <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
maybeShuffleBatchJoin :: String -> String -> String -> String -> String -> Tensor n t a 
maybeShuffleBatchJoin tensors_list batch_size capacity min_after_dequeue keep_input = TSym "tf.maybe_shuffle_batch_join" <+> TArgS "tensors_list" tensors_list <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "min_after_dequeue" min_after_dequeue <+> TArgS "keep_input" keep_input 

naturalExpDecay' :: String -> String -> String -> String -> String -> String -> Tensor n t a 
naturalExpDecay' learning_rate global_step decay_steps decay_rate staircase name = TSym "tf.natural_exp_decay" <+> TArgS "learning_rate" learning_rate <+> TArgS "global_step" global_step <+> TArgS "decay_steps" decay_steps <+> TArgS "decay_rate" decay_rate <+> TArgS "staircase" staircase <+> TArgS "name" name 
naturalExpDecay :: String -> String -> String -> String -> Tensor n t a 
naturalExpDecay learning_rate global_step decay_steps decay_rate = TSym "tf.natural_exp_decay" <+> TArgS "learning_rate" learning_rate <+> TArgS "global_step" global_step <+> TArgS "decay_steps" decay_steps <+> TArgS "decay_rate" decay_rate 

piecewiseConstant' :: Tensor n t a -> String -> String -> String -> Tensor n t a 
piecewiseConstant' x boundaries values name = TSym "tf.piecewise_constant" <+> TArgT "x" x <+> TArgS "boundaries" boundaries <+> TArgS "values" values <+> TArgS "name" name 
piecewiseConstant :: Tensor n t a -> String -> String -> Tensor n t a 
piecewiseConstant x boundaries values = TSym "tf.piecewise_constant" <+> TArgT "x" x <+> TArgS "boundaries" boundaries <+> TArgS "values" values 

polynomialDecay' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
polynomialDecay' learning_rate global_step decay_steps end_learning_rate power cycle name = TSym "tf.polynomial_decay" <+> TArgS "learning_rate" learning_rate <+> TArgS "global_step" global_step <+> TArgS "decay_steps" decay_steps <+> TArgS "end_learning_rate" end_learning_rate <+> TArgS "power" power <+> TArgS "cycle" cycle <+> TArgS "name" name 
polynomialDecay :: String -> String -> String -> Tensor n t a 
polynomialDecay learning_rate global_step decay_steps = TSym "tf.polynomial_decay" <+> TArgS "learning_rate" learning_rate <+> TArgS "global_step" global_step <+> TArgS "decay_steps" decay_steps 

rangeInputProducer' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
rangeInputProducer' limit num_epochs shuffle seed capacity shared_name name = TSym "tf.range_input_producer" <+> TArgS "limit" limit <+> TArgS "num_epochs" num_epochs <+> TArgS "shuffle" shuffle <+> TArgS "seed" seed <+> TArgS "capacity" capacity <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
rangeInputProducer :: String -> Tensor n t a 
rangeInputProducer limit = TSym "tf.range_input_producer" <+> TArgS "limit" limit 


replicaDeviceSetter :: Tensor n t a 
replicaDeviceSetter = TSym "tf.replica_device_setter" 

sdcaFprint' :: String -> String -> Tensor n t a 
sdcaFprint' input name = TSym "tf.sdca_fprint" <+> TArgS "input" input <+> TArgS "name" name 
sdcaFprint :: String -> Tensor n t a 
sdcaFprint input = TSym "tf.sdca_fprint" <+> TArgS "input" input 

sdcaOptimizer' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
sdcaOptimizer' sparse_example_indices sparse_feature_indices sparse_feature_values dense_features example_weights example_labels sparse_indices sparse_weights dense_weights example_state_data loss_type l1 l2 num_loss_partitions num_inner_iterations adaptative name = TSym "tf.sdca_optimizer" <+> TArgS "sparse_example_indices" sparse_example_indices <+> TArgS "sparse_feature_indices" sparse_feature_indices <+> TArgS "sparse_feature_values" sparse_feature_values <+> TArgS "dense_features" dense_features <+> TArgS "example_weights" example_weights <+> TArgS "example_labels" example_labels <+> TArgS "sparse_indices" sparse_indices <+> TArgS "sparse_weights" sparse_weights <+> TArgS "dense_weights" dense_weights <+> TArgS "example_state_data" example_state_data <+> TArgS "loss_type" loss_type <+> TArgS "l1" l1 <+> TArgS "l2" l2 <+> TArgS "num_loss_partitions" num_loss_partitions <+> TArgS "num_inner_iterations" num_inner_iterations <+> TArgS "adaptative" adaptative <+> TArgS "name" name 
sdcaOptimizer :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
sdcaOptimizer sparse_example_indices sparse_feature_indices sparse_feature_values dense_features example_weights example_labels sparse_indices sparse_weights dense_weights example_state_data loss_type l1 l2 num_loss_partitions num_inner_iterations = TSym "tf.sdca_optimizer" <+> TArgS "sparse_example_indices" sparse_example_indices <+> TArgS "sparse_feature_indices" sparse_feature_indices <+> TArgS "sparse_feature_values" sparse_feature_values <+> TArgS "dense_features" dense_features <+> TArgS "example_weights" example_weights <+> TArgS "example_labels" example_labels <+> TArgS "sparse_indices" sparse_indices <+> TArgS "sparse_weights" sparse_weights <+> TArgS "dense_weights" dense_weights <+> TArgS "example_state_data" example_state_data <+> TArgS "loss_type" loss_type <+> TArgS "l1" l1 <+> TArgS "l2" l2 <+> TArgS "num_loss_partitions" num_loss_partitions <+> TArgS "num_inner_iterations" num_inner_iterations 

sdcaShrinkL1' :: String -> String -> String -> String -> Tensor n t a 
sdcaShrinkL1' weights l1 l2 name = TSym "tf.sdca_shrink_l1" <+> TArgS "weights" weights <+> TArgS "l1" l1 <+> TArgS "l2" l2 <+> TArgS "name" name 
sdcaShrinkL1 :: String -> String -> String -> Tensor n t a 
sdcaShrinkL1 weights l1 l2 = TSym "tf.sdca_shrink_l1" <+> TArgS "weights" weights <+> TArgS "l1" l1 <+> TArgS "l2" l2 

shuffleBatch' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
shuffleBatch' tensors batch_size capacity min_after_dequeue num_threads seed enqueue_many shapes allow_smaller_final_batch shared_name name = TSym "tf.shuffle_batch" <+> TArgS "tensors" tensors <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "min_after_dequeue" min_after_dequeue <+> TArgS "num_threads" num_threads <+> TArgS "seed" seed <+> TArgS "enqueue_many" enqueue_many <+> TArgS "shapes" shapes <+> TArgS "allow_smaller_final_batch" allow_smaller_final_batch <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
shuffleBatch :: String -> String -> String -> String -> Tensor n t a 
shuffleBatch tensors batch_size capacity min_after_dequeue = TSym "tf.shuffle_batch" <+> TArgS "tensors" tensors <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "min_after_dequeue" min_after_dequeue 

shuffleBatchJoin' :: String -> String -> String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
shuffleBatchJoin' tensors_list batch_size capacity min_after_dequeue seed enqueue_many shapes allow_smaller_final_batch shared_name name = TSym "tf.shuffle_batch_join" <+> TArgS "tensors_list" tensors_list <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "min_after_dequeue" min_after_dequeue <+> TArgS "seed" seed <+> TArgS "enqueue_many" enqueue_many <+> TArgS "shapes" shapes <+> TArgS "allow_smaller_final_batch" allow_smaller_final_batch <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
shuffleBatchJoin :: String -> String -> String -> String -> Tensor n t a 
shuffleBatchJoin tensors_list batch_size capacity min_after_dequeue = TSym "tf.shuffle_batch_join" <+> TArgS "tensors_list" tensors_list <+> TArgS "batch_size" batch_size <+> TArgS "capacity" capacity <+> TArgS "min_after_dequeue" min_after_dequeue 

sliceInputProducer' :: String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
sliceInputProducer' tensor_list num_epochs shuffle seed capacity shared_name name = TSym "tf.slice_input_producer" <+> TArgS "tensor_list" tensor_list <+> TArgS "num_epochs" num_epochs <+> TArgS "shuffle" shuffle <+> TArgS "seed" seed <+> TArgS "capacity" capacity <+> TArgS "shared_name" shared_name <+> TArgS "name" name 
sliceInputProducer :: String -> Tensor n t a 
sliceInputProducer tensor_list = TSym "tf.slice_input_producer" <+> TArgS "tensor_list" tensor_list 


startQueueRunners :: Tensor n t a 
startQueueRunners = TSym "tf.start_queue_runners" 

stringInputProducer' :: String -> String -> String -> String -> String -> String -> String -> String -> Tensor n t a 
stringInputProducer' string_tensor num_epochs shuffle seed capacity shared_name name cancel_op = TSym "tf.string_input_producer" <+> TArgS "string_tensor" string_tensor <+> TArgS "num_epochs" num_epochs <+> TArgS "shuffle" shuffle <+> TArgS "seed" seed <+> TArgS "capacity" capacity <+> TArgS "shared_name" shared_name <+> TArgS "name" name <+> TArgS "cancel_op" cancel_op 
stringInputProducer :: String -> Tensor n t a 
stringInputProducer string_tensor = TSym "tf.string_input_producer" <+> TArgS "string_tensor" string_tensor 


summaryIterator :: String -> Tensor n t a 
summaryIterator path = TSym "tf.summary_iterator" <+> TArgS "path" path 

updateCheckpointState' :: String -> String -> String -> String -> Tensor n t a 
updateCheckpointState' save_dir model_checkpoint_path all_model_checkpoint_paths latest_filename = TSym "tf.update_checkpoint_state" <+> TArgS "save_dir" save_dir <+> TArgS "model_checkpoint_path" model_checkpoint_path <+> TArgS "all_model_checkpoint_paths" all_model_checkpoint_paths <+> TArgS "latest_filename" latest_filename 
updateCheckpointState :: String -> String -> Tensor n t a 
updateCheckpointState save_dir model_checkpoint_path = TSym "tf.update_checkpoint_state" <+> TArgS "save_dir" save_dir <+> TArgS "model_checkpoint_path" model_checkpoint_path 

writeGraph' :: String -> String -> String -> String -> Tensor n t a 
writeGraph' graph_or_graph_def logdir name as_text = TSym "tf.write_graph" <+> TArgS "graph_or_graph_def" graph_or_graph_def <+> TArgS "logdir" logdir <+> TArgS "name" name <+> TArgS "as_text" as_text 
writeGraph :: String -> String -> String -> Tensor n t a 
writeGraph graph_or_graph_def logdir name = TSym "tf.write_graph" <+> TArgS "graph_or_graph_def" graph_or_graph_def <+> TArgS "logdir" logdir <+> TArgS "name" name 

