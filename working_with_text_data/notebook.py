import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    from urllib import request
    import re
    import tiktoken
    import torch
    import importlib
    from torch.utils.data import Dataset, DataLoader
    return DataLoader, Dataset, importlib, mo, os, re, request, tiktoken, torch


@app.cell
def _(os, request):
    file_path = os.path.join(os.path.dirname(__file__), "the-verdict.txt")
    if not os.path.exists(file_path):
        url = (
            "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
            "/refs/heads/main/ch02/01_main-chapter-code/the-verdict.txt"
        )
        request.urlretrieve(url, file_path)
    return (file_path,)


@app.cell
def _(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    return (raw_text,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""get tokens from dataset""")
    return


@app.cell
def _(re):
    text = "Hello, world. This, is a test."
    result = re.split(r"(\s)", text)

    print(result)
    return result, text


@app.cell
def _(re, text):
    print(re.split(r"([,.]|\s)", text))
    return


@app.cell
def _(result):
    print([item for item in result if item.strip()])
    return


@app.cell
def _(mo):
    mo.md(r"""tokenizing text""")
    return


@app.cell
def _(re):
    def regex_tokenizer():
        text = "Hello, world. Is this-- a test?"

        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [item.strip() for item in result if item.strip()]
        return print(result)

    regex_tokenizer()
    return


@app.cell
def _(raw_text, re):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])
    return (preprocessed,)


@app.cell
def _(preprocessed):
    print(len(preprocessed))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""tokens -> token IDs""")
    return


@app.cell
def _(preprocessed):
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)

    print(vocab_size)
    return (all_words,)


@app.cell
def _(all_words):
    vocab = {token:integer for integer,token in enumerate(all_words)}
    return (vocab,)


@app.cell
def _(vocab):
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 50:
            break
    return


@app.cell
def _(re):
    class SimpleTokenizerV1:
        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i:s for s,i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

            preprocessed = [
                item.strip() for item in preprocessed if item.strip()
            ]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            # Replace spaces before the specified punctuations
            text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
            return text
    return (SimpleTokenizerV1,)


@app.cell
def _(SimpleTokenizerV1, vocab):
    tokenizer = SimpleTokenizerV1(vocab)
    text_to_tokenize = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text_to_tokenize)
    print(ids)
    return ids, text_to_tokenize, tokenizer


@app.cell
def _(ids, tokenizer):
    tokenizer.decode(ids)
    return


@app.cell
def _(text_to_tokenize, tokenizer):
    tokenizer.decode(tokenizer.encode(text_to_tokenize))
    return


@app.cell
def _(mo):
    mo.md(r"""special ctx tokens""")
    return


@app.cell
def _(SimpleTokenizerV1, vocab):
    def _():
        tokenizer = SimpleTokenizerV1(vocab)
        text = "Hello, do you like tea. Is this-- a test?"
        return tokenizer.encode(text)
    _()
    return


@app.cell
def _(preprocessed):
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])

    new_vocab = {token:integer for integer,token in enumerate(all_tokens)}
    return (new_vocab,)


@app.cell
def _(new_vocab):
    len(new_vocab.items())
    return


@app.cell
def _(new_vocab):
    def _():
        for i, item in enumerate(list(new_vocab.items())[-5:]):
            return print(item)
    _()
    return


@app.cell
def _(new_vocab, re):
    class SimpleTokenizerV2:
        def __init__(self, vocab):
            self.str_to_int = new_vocab
            self.int_to_str = { i:s for s,i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            preprocessed = [
                item if item in self.str_to_int 
                else "<|unk|>" for item in preprocessed
            ]

            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            # Replace spaces before the specified punctuations
            text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
            return text
    return (SimpleTokenizerV2,)


@app.cell
def _(SimpleTokenizerV2, new_vocab):
    tokenizerV2 = SimpleTokenizerV2(new_vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    new_text_to_tokenize = " <|endoftext|> ".join((text1, text2))

    print(new_text_to_tokenize)
    return new_text_to_tokenize, tokenizerV2


@app.cell
def _(new_text_to_tokenize, tokenizerV2):
    tokenizerV2.encode(new_text_to_tokenize)
    return


@app.cell
def _(new_text_to_tokenize, tokenizerV2):
    tokenizerV2.decode(tokenizerV2.encode(new_text_to_tokenize))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""BPE""")
    return


@app.cell
def _(importlib):
    print("tiktoken version:", importlib.metadata.version("tiktoken"))
    return


@app.cell
def _(tiktoken):
    gpt2_tokenizer = tiktoken.get_encoding("gpt2")
    return (gpt2_tokenizer,)


@app.cell
def _(gpt2_tokenizer):
    gpt2_text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
         "of someunknownPlace."
    )
    integers = gpt2_tokenizer.encode(gpt2_text, allowed_special={"<|endoftext|>"})
    print (integers)
    strings = gpt2_tokenizer.decode(integers)
    print (strings)
    return


@app.cell
def _(gpt2_tokenizer, raw_text):
    enc_text = gpt2_tokenizer.encode(raw_text)
    print(len(enc_text))
    return (enc_text,)


@app.cell
def _(enc_text):
    enc_sample = enc_text[50:]
    return (enc_sample,)


@app.cell
def _(enc_sample):
    context_size = 4

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    print(f"x: {x}")
    print(f"y:      {y}")
    return (context_size,)


@app.cell
def _(context_size, enc_sample):
    for _ in range(1, context_size+1):
        _context = enc_sample[:_]
        _desired = enc_sample[_]

        print(_context, "---->", _desired)
    return


@app.cell
def _(context_size, enc_sample, gpt2_tokenizer):
    for _ in range(1, context_size+1):
        context = enc_sample[:_]
        desired = enc_sample[_]
        print(gpt2_tokenizer.decode(context), "---->", 

    gpt2_tokenizer.decode([desired]))
    return


@app.cell
def _(torch):
    torch.__version__
    return


@app.cell
def _(Dataset, torch):
    class GPTDatasetV1(Dataset):
        def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids = []
            self.target_ids = []

            # Tokenize the entire text
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
            assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

            # Use a sliding window to chunk the book into overlapping sequences of max_length
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i:i + max_length]
                target_chunk = token_ids[i + 1: i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]
    return (GPTDatasetV1,)


@app.cell
def _(DataLoader, GPTDatasetV1, tiktoken):
    def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                             stride=128, shuffle=True, drop_last=True,
                             num_workers=0):

        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

        return dataloader
    return (create_dataloader_v1,)


@app.cell
def _(mo):
    mo.md(r"""test the dataloader with a batch size of 1 for an LLM with a context size of 4""")
    return


@app.cell
def _(create_dataloader_v1, raw_text):
    _dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    _data_iter = iter(_dataloader)
    first_batch = next(_data_iter)
    print(first_batch)
    second_batch = next(_data_iter)
    print(second_batch)
    return


@app.cell
def _(create_dataloader_v1, raw_text):
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""token emb""")
    return


@app.cell
def _(torch):
    _input_ids = torch.tensor([2, 3, 5, 1])
    _vocab_size = 6
    _output_dim = 3

    torch.manual_seed(123)
    _embedding_layer = torch.nn.Embedding(_vocab_size, _output_dim)
    print(_embedding_layer.weight)
    print("\n")
    # convert a token with id 3 into a 3D vector
    print("token id 3 -> 3D vector")
    print(_embedding_layer(torch.tensor([3])))

    # the above is the 4th row in the embedding_layer weight matrix
    print("\n")
    print("embed all four input_ids values above")
    print(_embedding_layer(_input_ids))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""enc word pos""")
    return


@app.cell
def _(create_dataloader_v1, raw_text, torch):
    def input_to_256D_vec():
        vocab_size = 50257
        output_dim = 256
        token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

        max_length = 4
        dataloader = create_dataloader_v1(
            raw_text, batch_size=8, max_length=max_length,
            stride=max_length, shuffle=False
        )
        data_iter = iter(dataloader)
        inputs, targets = next(data_iter)
        print("Token IDs:\n", inputs)
        print("\nInputs shape:\n", inputs.shape)
        token_embeddings = token_embedding_layer(inputs)
        print(token_embeddings.shape)
        return token_embeddings
    input_to_256D_vec()
    return (input_to_256D_vec,)


@app.cell
def _(input_to_256D_vec, torch):
    embeddings = input_to_256D_vec()
    pos_embedding_layer = torch.nn.Embedding(4, 256) # ctx len. output dim
    pos_embeddings = pos_embedding_layer(torch.arange(4))
    print(pos_embeddings.shape)
    input_embeddings = embeddings + pos_embeddings
    print(input_embeddings.shape)
    return


if __name__ == "__main__":
    app.run()
