from . interface import init_dd_state, dd_state
# TODO: fix whatever this is
# Traceback (most recent call last):
#   File "/home/mario/work/dev/FunctionalScenes/test/graphics/mitsuba_volume.py", line 9, in <module>
#     from functional_scenes.render.interface import (initialize_scene,
#   File "/home/mario/work/dev/FunctionalScenes/functional_scenes/__init__.py", line 5, in <module>
#     from . og_proposal import init_dd_state, dd_state
#   File "/home/mario/work/dev/FunctionalScenes/functional_scenes/og_proposal/__init__.py", line 2, in <module>
#     from . dataset import load_og
#   File "/home/mario/work/dev/FunctionalScenes/functional_scenes/og_proposal/dataset.py", line 9, in <module>
#     from ffcv.writer import DatasetWriter
#   File "/home/mario/work/dev/FunctionalScenes/env.d/pyenv/lib/python3.9/site-packages/ffcv/__init__.py", line 1, in <module>
#     from .loader import Loader
#   File "/home/mario/work/dev/FunctionalScenes/env.d/pyenv/lib/python3.9/site-packages/ffcv/loader/__init__.py", line 1, in <module>
#     from .loader import Loader, OrderOption
#   File "/home/mario/work/dev/FunctionalScenes/env.d/pyenv/lib/python3.9/site-packages/ffcv/loader/loader.py", line 14, in <module>
#     from ffcv.fields.base import Field
#   File "/home/mario/work/dev/FunctionalScenes/env.d/pyenv/lib/python3.9/site-packages/ffcv/fields/__init__.py", line 3, in <module>
#     from .rgb_image import RGBImageField
#   File "/home/mario/work/dev/FunctionalScenes/env.d/pyenv/lib/python3.9/site-packages/ffcv/fields/rgb_image.py", line 5, in <module>
#     import cv2
#   File "/home/mario/work/dev/FunctionalScenes/env.d/pyenv/lib/python3.9/site-packages/cv2/__init__.py", line 181, in <module>
#     bootstrap()
#   File "/home/mario/work/dev/FunctionalScenes/env.d/pyenv/lib/python3.9/site-packages/cv2/__init__.py", line 153, in bootstrap
#     native_module = importlib.import_module("cv2")
#   File "/usr/lib/python3.9/importlib/__init__.py", line 127, in import_module
#     return _bootstrap._gcd_import(name[level:], package, level)
# ImportError: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by /.singularity.d/libs/libGLdispatch.so.0)
# from . dataset import load_og
