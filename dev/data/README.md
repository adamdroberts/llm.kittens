# dev/data organization

The idea is that each dataset has a .py file here in the root of `dev/data`, and each dataset then creates a directory here, and writes and caches anything inside that directory. So for example:

- running `python tinystories.py` will create a directory `tinystories` with its .bin files inside it
- running `python tinyshakespeare.py` will create a directory `tinyshakespeare` with its .bin files inside it

And so on. This way we can nicely organize multiple datasets here, share common utilities between them, and then point the .py/.c code in the root of the project accordingly to these.

Note: we support "gpt-2" and "llama" (llama 3 in particular) models and the above scripts will tokenize gpt-2 by default. Use `--model_desc llama-3` to write Llama-3 token files. For HellaSwag, GPT-2 writes `hellaswag_val.bin`; Llama-3 writes `hellaswag_val_llama3.bin`.

After preprocessing, run `python ../validate_data_artifacts.py` from this
directory or `python dev/validate_data_artifacts.py` from the repo root. The
validator checks train/eval header magic, version, exact payload sizes, sampled
token ranges, and the full HellaSwag-style eval stream without CUDA.
