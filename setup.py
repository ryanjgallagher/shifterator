import setuptools

with open('README.md', 'r') as f:
	long_description = f.read()

setuptools.setup(
	name='shifterator',
	version='0.1.3',
	author='Ryan J. Gallagher',
	author_email='gallagher.r@northeastern.edu',
	keywords=['natural language processing', 'sentiment analysis', 'information theory', 'computational social socience' , 'digital humanities', 'text analysis', 'text as data', 'data visualization', 'data viz'],
	description='Interpretable data visualizations for understanding how texts differ at the word level',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/ryanjgallagher/shifterator',
	packages=setuptools.find_packages(),
    include_package_data=True,
	classifiers=[
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'License :: OSI Approved :: Apache Software License',
		'Operating System :: OS Independent',
	],
)
