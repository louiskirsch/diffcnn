from distutils.core import setup
from pip.req import parse_requirements

setup(
  name='mcnn',
  version='1.0',
  packages=['mcnn'],
  scripts=['mcnn-cli'],
  url='https://github.com/timediv/mutating-cnn',
  license='Apache License 2.0',
  author='Louis Kirsch',
  author_email='mail@louiskirsch.com',
  description='Making time series classification architectures differentiable',
  install_requires=[str(r.req) for r in parse_requirements("requirements.txt", session=False)]
)
