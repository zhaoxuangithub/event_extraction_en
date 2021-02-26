"""
flask manage
"""
import importlib
import pkgutil
import os
import sys
import json
import time

from flask import Flask, request, Response
# from allennlp.commands.predict import Predict
from allennlp.commands import create_parser, predict
from allennlp.common.plugins import import_plugins
from allennlp.common.util import import_module_and_submodules, push_python_path
import torch
from overrides import overrides
from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import SentenceTaggerPredictor
from allennlp.predictors import TextClassifierPredictor
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

sys.path.append(os.path.abspath('.'))
# print(sys.path)
from dygie.predictors.format_dataset_util import init_spacy_model, format_dataset_new, read_pred_json_format
from dygie.predictors.dygie import DyGIEPredictor

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

spacy_model = None
predict_manager = None
cuda_place = 1
current_path = os.getcwd()


def release_torch_gpu_memory():
    torch.cuda.empty_cache()
    time.sleep(5)


def import_module_and_submodules_new(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    """
    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path("."):
        # Import at top level
        try:
            module = importlib.import_module(package_name)
        except Exception as e:
            print(e)
            module = None
        if module:
            path = getattr(module, "__path__", [])
            path_string = "" if not path else path[0]
    
            # walk_packages only finds immediate children, so need to recurse.
            for module_finder, name, _ in pkgutil.walk_packages(path):
                # Sometimes when you import third-party libraries that are on your path,
                # `pkgutil.walk_packages` returns those too, so we need to skip them.
                if path_string and module_finder.path != path_string:
                    continue
                subpackage = f"{package_name}.{name}"
                import_module_and_submodules(subpackage)


def init_load_model():
    """
    The [`run`](./train.md#run) command only knows about the registered classes in the ``allennlp``
    codebase. In particular, once you start creating your own `Model` s and so forth, it won't
    work for them, unless you use the ``--include-package`` flag or you make your code available
    as a plugin (see [`plugins`](./plugins.md)).
    """
    global predict_manager, cuda_place
    import_plugins()

    parser = create_parser("allennlp")
    args = parser.parse_args()
    args.archive_file = r"models/ace05_event/model.tar.gz"
    args.cuda_device = cuda_place
    args.input_file = r"data/ace-event/processed-data/default-settings/json/test.json"
    args.output_file = r"predictions/event_predict.json"
    args.use_dataset_reader = True
    args.include_package = "dygie"
    args.predictor = "dygie"
    args.weights_file = None
    args.overrides = ''
    args.dataset_reader_choice = 'validation'
    args.batch_size = 1
    args.silent = False
    
    # print(dir(args))
    # print(args.include_package)
    # for package_name in args.include_package:
    #     import_module_and_submodules_new(package_name)
    import_module_and_submodules_new('dygie')
    import_module_and_submodules_new('dygie.predictors.dygie')
    import_module_and_submodules_new('dygie.models.dygie')
    # predictor = DyGIEPredictor.from_path(os.path.join(current_path, 'models/ace05_event/model.tar.gz'), predictor_name="dygie")
    predictor = predict._get_predictor(args)
    
    predict_manager = predict._PredictManager(
        predictor,
        args.input_file,
        args.output_file,
        args.batch_size,
        not args.silent,
        args.use_dataset_reader,
    )
    
    # if "func" in dir(args):
    #     pass
    # else:
    #     print("no func")
    #     parser.print_help()
        
        
@app.route('/event_extraction_en', methods=['POST'])
def rel_extract():
    # release_torch_gpu_memory()
    try:
        with torch.no_grad():
            texts = request.json['content']
            # 格式化数据并存入指定的test.json文件中
            content_dict = format_dataset_new(spacy_model, texts,
                                              os.path.join(current_path, "data/ace-event/processed-data/default-settings/json/test.json"))
        
            if content_dict:
                # 有可以预测的内容
                if predict_manager:
                    if predict_manager._output_file is not None:
                        predict_manager._output_file.close()
                    predict_manager._output_file = open(r"predictions/event_predict.json", "w")
                    predict_manager.run()
                else:
                    print("predict_manager is none")
            
            result_dict = {'content': read_pred_json_format(content_dict)}
            result_json = json.dumps(result_dict, ensure_ascii=False)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("GPU 显存不足！")
            if hasattr(torch.cuda, "empty_cache"):
                release_torch_gpu_memory()
            else:
                print(e)
        result_json = json.dumps({}, ensure_ascii=False)
    # 模型最大gpu显存占用获取
    # gpu_mem = torch.cuda.max_memory_reserved(cuda_place)
    # 模型当前gpu显存占用获取
    gpu_mem = torch.cuda.memory_reserved(cuda_place)
    print("-------------test_gpu_memory------------", gpu_mem)
    if gpu_mem > 5242880000:
        # over 5G release gpu memory
        release_torch_gpu_memory()
    gpu_mem = torch.cuda.max_memory_reserved(cuda_place)
    print("-------------test_gpu_memory------------", gpu_mem)
    if gpu_mem > 5242880000:
        # over 5G release gpu memory
        release_torch_gpu_memory()
        
    # release_torch_gpu_memory()
    return Response(result_json, mimetype='application/json')


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        # 与args冲突，尚未解决目前不支持指定cuda_place
        int_place = -1
        try:
            int_place = int(sys.argv[1])
        except Exception as e:
            print("cuda_place argument input error ues default place 0 !!!")
        if int_place != -1:
            cuda_place = int_place
    print('----------start server please wait ...-------------------------')
    # 初始化spacy
    spacy_model = init_spacy_model()
    # 初始化 模型
    init_load_model()
    # app.run(host='0.0.0.0', port=4445)
    app.run(host='0.0.0.0', port=44445)
