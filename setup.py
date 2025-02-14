from setuptools import setup, find_packages

with open('requirement.txt') as f:
    required_packages = f.read().splitlines()

setup(
    name='auto_flow',
    version='0.1.1',
    packages=find_packages(where='src'),  # 指定包的位置
    package_dir={'': 'src'},
    # package_data={"xxx.config": ["*.json"]},
    install_requires=required_packages,
    author='zhiguo',
    author_email='zhiguoxu2004@163.com',
    description='auto_flow',
    long_description="",
    long_description_content_type='text/markdown',
    url='https://github.com/xxx',
    classifiers=[]
)
