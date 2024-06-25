from setuptools import setup, find_packages

setup(
    name='pricing',
    version='0.1.0',
    description='Optimal Pricing',
    author='Alex Monras',
    author_email='alexmonrasblasi@gmail.com',
    url='https://github.com/amonras/optimal-pricing',
    packages=['pricing', 'pricing.models'],
    package_dir={'pricing': 'pricing'},
    install_requires=[
        'pandas',
        'scikit-learn',
        'plotly',
        'matplotlib',
        'seaborn',
        'scipy',
        'numpy',
        'jupyter',
        'pyarrow',
        'openpyxl',
        'pytest',
        'tqdm',
        'dataframe_image'
    ],
)