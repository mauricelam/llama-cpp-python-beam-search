# llama-cpp-python-beam-search
Beam search implementation in llama-cpp-python

`beam_search.py` and `beam_search_lowlevel.py` perform the same functions, so you would only choose one or the other to use.

- `beam_search.py` uses the high-level API by llama-cpp-python, but requires using the internal APIs and may therefore be unstable.
- `beam_search_lowlevel.py` uses the low-level API, which is almost one-to-one to the llama.cpp API. It does not use any internal APIs and should be more stable. This code should be easy to port to C++ as well.

## Usage

See https://colab.research.google.com/drive/1jVyLsfHlUhsnl1p_pMx-eUudrTeiwpUU?usp=sharing for usage example.

## Links

- https://huggingface.co/blog/how-to-generate
