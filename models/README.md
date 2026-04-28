# Models

The fine-tuned TrOCR weights live in `models/trocr_model/`. The folder
should contain the usual Hugging Face files:

```
trocr_model/
├── config.json
├── generation_config.json
├── preprocessor_config.json
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── model.safetensors      # ~1.3 GB -- the actual weights
```

If `model.safetensors` is missing or the folder is empty, `app.py` falls
back to the stock `microsoft/trocr-base-handwritten` model from the Hugging
Face Hub on first run. The UI will say
*"base pretrained (no fine-tuned checkpoint found)"* so you know.

## Reproducing the fine-tune

Run the training notebook at the repo root:

```
Final_Code.ipynb
```

It expects the dataset to be present (see `../data/README.md`) and writes
its best checkpoint back into this directory.
