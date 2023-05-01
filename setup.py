import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name="gym-classics",
    version="1.0.0",
    author="Brett Daley",
    author_email="brett.daley@ualberta.ca",
    description="Classic environments for reinforcement learning and dynamic"
                " programming, implemented in OpenAI Gym and Gymnasium.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brett-daley/gym-classics",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
