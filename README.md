# Sparse Transcoder

**Work in progress, feedback and contributions welcome!**

Visualizing high-dimensional data is hard. But then, natural images are high-dimensional, and we can easily understand those? Here we implement *sparse transcoding*, a method to translate arbitrary high-dimensional data into natural images so we can see the structure in them better.

The basic steps of our method are simple:

1. Train a [sparse auto-encoder](https://transformer-circuits.pub/2023/monosemantic-features/index.html) for the target data (the "encoder").
2. Train a sparse auto-encoder for natural image patches (the "decoder").
3. Visualize by encoding with the encoder and decoding with the decoder ("transcoding").

## Installation

Clone the repo and install the package in a new virtual environment

```bash
git clone https://github.com/clane9/sparse-transcoder.git
cd sparse-transcoder

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install --no-deps -e .
```

## Inspiration

- [Sparse coding](http://www.scholarpedia.org/article/Sparse_coding)
- [Sparse autoencoders for interpretability](https://transformer-circuits.pub/2023/monosemantic-features/index.html)
- [Multi-layer sparse modeling](https://drive.google.com/file/d/1Vxhe1ikRdcsoiwjh5ns0knMPLP1YzZrS/view)

## Contribute

If you'd like to contribute, please feel free fork the repo and start a conversation in our [issues](https://github.com/clane9/sparse-transcoder/issues).
