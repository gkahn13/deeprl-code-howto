import importlib.util


def import_config(config_fname):
    assert config_fname.endswith('.py')
    spec = importlib.util.spec_from_file_location('config', config_fname)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.params
