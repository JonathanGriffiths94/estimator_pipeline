from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['sh']

setup(name='trainer',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      description='wafer defect classifier tensorflow estimator cnn',
      author='Jonathan Griffiths',
      author_email='jonathan.griffiths@maiple.com',
      license='Maiple.ltd',
      zip_safe=False)