def dump_code(framework, network_filepath, weight_filepath, dump_filepath):
    if network_filepath.endswith('.py'):
        network_filepath = network_filepath[:-3]
    MainModel = __import__(network_filepath)
    if framework == 'caffe':
        from mmdnn.conversion.caffe.saver import save_model
    elif framework == 'cntk':
        from mmdnn.conversion.cntk.saver import save_model
    elif framework == 'keras':
        from mmdnn.conversion.keras.saver import save_model
    elif framework == 'mxnet':
        from mmdnn.conversion.mxnet.saver import save_model
    elif framework == 'pytorch':
        from mmdnn.conversion.pytorch.saver import save_model
    elif framework == 'tensorflow':
        from mmdnn.conversion.tensorflow.saver import save_model
    else:
        raise NotImplementedError("{} saver is not finished yet.".format(framework))
    save_model(MainModel, network_filepath, weight_filepath, dump_filepath)
