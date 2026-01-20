import importlib
import traceback

try:
    importlib.import_module('api.app')
    print('imported api.app successfully')
except Exception:
    traceback.print_exc()
