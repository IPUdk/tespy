from setuptools import setup
import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='TESPy',
      version='0.2.0 dev',
      description='Thermal Engineering Systems in Python (TESPy)',
      url='http://github.com/oemof/tespy',
      author='Francesco Witte',
      author_email='francesco.witte@web.de',
      long_description=read('README.rst'),
      license='MIT',
      packages=['tespy', 'tespy.components', 'tespy.data', 'tespy.networks',
                'tespy.tools'],
      data_files=[('data', ['data/*.json'])],
      python_requires='>=3',
      install_requires=['CoolProp >= 6.0.0',
                        'numpy >= 1.13.3',
                        'pandas >= 0.19.2',
                        'scipy >= 0.19.1',
                        'tabulate >= 0.8.2'])
