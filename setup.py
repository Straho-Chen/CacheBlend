# setup.py（放在 repo 根）
from setuptools import setup, find_packages

setup(
    name="cacheblend-utils",           # 发布/安装包的名称（随你取名，不一定要和目录同名）
    version="0.0.1",
    description="Small utilities used by CacheBlend tests",
    packages=find_packages(include=["utils", "utils.*"]),  # 安装 utils 包
    include_package_data=True,
    install_requires=[
    ],
)