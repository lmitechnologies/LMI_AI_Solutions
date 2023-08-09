#!/bin/bash

usage_and_exit() {
    echo ====================================================
    echo main.sh [options]
    echo -a action to perform
    echo    train    train anomaly model
    echo    test     test trained model
    echo    convert  convert trained model to trt engine
    echo -o output directory for action results
    echo -d train data path, only required for train
    echo -t test data path, only required for test
    echo -e the folder that contains model.engine to test, defaults to latest folder in output/engines
    echo -m the model type to train: padim as default or patchcore
    echo -x the folder that contains model.onnx to be converted to trt engine, defaults to the latest in output/results/model/trained models
    echo examples:
    echo -------- train -----
    echo bash main.sh -a train -d /path/to/train/data -t /optionally/path/to/test/data -o /path/to/outdir
    echo -------- convert -----
    echo bash main.sh -a convert -x /path/to/onnx -o /path/to/outdir 
    echo -------- test -----
    echo bash main.sh -a test -e /path/to/trt/engine -t /path/to/test/data -o /path/to/outdir
    echo =====================================================
    exit 0
}

model=padim
while getopts "h?a:d:t:o:x:e:m:" opt; do
  case "$opt" in
    h|\?)
    echo opt $opt
      usage_and_exit
      ;;
    a)  action=$OPTARG
      ;;
    d)  train_data=$OPTARG
      ;;
    t)  test_data=$OPTARG
      ;;
    o)  out_dir=$OPTARG
      ;;
    m)  model=$OPTARG
      ;;
    x)  onnx_dir=$OPTARG
      ;;
    e)  engine_dir=$OPTARG
      ;;
  esac
done

if [[ -z $action || -z $out_dir ]]; then
  usage_and_exit
fi
if [[ $action == "train" && -z $train_data ]]; then
  echo no train data for train action
  usage_and_exit
fi
if [[ $action == "test" && -z $test_data ]]; then
  echo no test data for test action
  usage_and_exit
fi

image_name=anomalib_runner:latest
docker build -f docker/dockerfile.$(uname -p) . -t $image_name

if [[ $out_dir != /* ]]; then
  out_dir=$(pwd)/$out_dir
fi
if [[ ! -d $out_dir ]]; then
  mkdir -p $out_dir
fi

docker_run_cmd="docker run --name anomalib_runner_1 --runtime nvidia --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --rm -v $(pwd):/app/ws -v $out_dir:/app/out"
if [[ $train_data ]]; then
  if [[ $train_data != /* ]]; then
    train_data=$(pwd)/$train_data
  fi
  docker_run_cmd="${docker_run_cmd} -v $train_data:/app/data/good"
fi
if [[ $test_data ]]; then
  if [[ $test_data != /* ]]; then
    test_data=$(pwd)/$test_data
  fi
  docker_run_cmd="${docker_run_cmd} -v $test_data:/app/data/test"
fi
if [[ $onnx_dir ]]; then
  if [[ $onnx_dir != /* ]]; then
    onnx_dir=$(pwd)/$onnx_dir
  fi
  docker_run_cmd="${docker_run_cmd} -v $onnx_dir:/app/mounted/onnx"
fi
if [[ $engine_dir ]]; then
  if [[ $engine_dir != /* ]]; then
    engine_dir=$(pwd)/$engine_dir
  fi
  docker_run_cmd="${docker_run_cmd} -v $engine_dir:/app/mounted/engine"
fi

docker_run_cmd="${docker_run_cmd} $image_name bash /app/ws/internal_runner.sh $action $model"

echo docker_run_cmd=$docker_run_cmd
$docker_run_cmd