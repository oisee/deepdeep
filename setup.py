from setuptools import setup, find_packages

with open("CLAUDE.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spectrumAI",
    version="0.1.0",
    author="SpectrumAI Team",
    description="Next-Generation ZX Spectrum Image Converter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.24",
        "opencv-python>=4.8",
        "Pillow>=10.0",
        "scipy>=1.10",
        "numba>=0.57",
    ],
    extras_require={
        "ml": [
            "transformers>=4.30",
            "kornia>=0.7",
            "ultralytics>=8.0",
            "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git",
        ],
        "ui": [
            "gradio>=3.35",
            "PyQt6>=6.5",
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "black>=23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spectrumAI=spectrumAI.cli:main",
        ],
    },
)