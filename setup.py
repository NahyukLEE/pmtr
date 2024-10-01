from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='pmtr',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='pmtr.ext',
            sources=[
                'common/extensions/extra/cloud/cloud.cpp',
                'common/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'common/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'common/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'common/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'common/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)