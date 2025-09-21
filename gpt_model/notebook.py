import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import tiktoken
    import matplotlib.pyplot as plt
    from attention_mechanisms import MultiHeadAttention
    return MultiHeadAttention, mo, nn, plt, tiktoken, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""impl LLM arch""")
    return


@app.cell
def _():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }
    return (GPT_CONFIG_124M,)


@app.cell
def _(nn, torch):
    class DummyGPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            # Use a placeholder for TransformerBlock
            self.trf_blocks = nn.Sequential(
                *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )

            # Use a placeholder for LayerNorm
            self.final_norm = DummyLayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits

    class DummyTransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            # A simple placeholder

        def forward(self, x):
            # This block does nothing and just returns its input.
            return x

    class DummyLayerNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            super().__init__()
            # The parameters here are just to mimic the LayerNorm interface.

        def forward(self, x):
            # This layer does nothing and just returns its input.
            return x
    return (DummyGPTModel,)


@app.cell
def _(tiktoken, torch):
    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print(batch)
    return batch, tokenizer


@app.cell
def _(DummyGPTModel, GPT_CONFIG_124M, batch, torch):
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""normalizing activations with LayerNorm""")
    return


@app.cell
def _(nn, torch):
    torch.manual_seed(123)

    # create 2 training examples with 5 dimensions (features) each
    batch_example = torch.randn(2, 5)

    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)
    return batch_example, out


@app.cell
def _(out, torch):
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)

    out_norm = (out - mean) / torch.sqrt(var)
    print("Normalized layer outputs:\n", out_norm)

    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)
    return


@app.cell
def _(nn, torch):
    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift
    return (LayerNorm,)


@app.cell
def _(LayerNorm, batch_example):
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    _mean = out_ln.mean(dim=-1, keepdim=True)
    _var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    print("Mean:\n", _mean)
    print("Variance:\n", _var)
    return


@app.cell
def _(mo):
    mo.md(r"""FFN with GELU""")
    return


@app.cell
def _(nn, torch):
    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return (
                0.5
                * x
                * (
                    1
                    + torch.tanh(
                        torch.sqrt(torch.tensor(2.0 / torch.pi))
                        * (x + 0.044715 * torch.pow(x, 3))
                    )
                )
            )
    return (GELU,)


@app.cell
def _(GELU, nn, plt, torch):
    gelu, relu = GELU(), nn.ReLU()

    # sample data
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(GELU, nn):
    class FeedForward(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            )

        def forward(self, x):
            return self.layers(x)
    return (FeedForward,)


@app.cell
def _(GPT_CONFIG_124M):
    print(GPT_CONFIG_124M["emb_dim"])
    return


@app.cell
def _(FeedForward, GPT_CONFIG_124M, torch):
    ffn = FeedForward(GPT_CONFIG_124M)

    # input shape: [batch_size, num_token, emb_size]
    _x = torch.rand(2, 3, 768)
    _out = ffn(_x)
    print(_out.shape)
    return


@app.cell
def _(GELU, nn, torch):
    class ExampleDeepNeuralNetwork(nn.Module):
        def __init__(self, layer_sizes, use_shortcut):
            super().__init__()
            self.use_shortcut = use_shortcut
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                    nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
                ]
            )

        def forward(self, x):
            for layer in self.layers:
                # Compute the output of the current layer
                layer_output = layer(x)
                # Check if shortcut can be applied
                if self.use_shortcut and x.shape == layer_output.shape:
                    x = x + layer_output
                else:
                    x = layer_output
            return x

    def print_gradients(model, x):
        # Forward pass
        output = model(x)
        target = torch.tensor([[0.0]])

        # Calculate loss based on how close the target
        # and output are
        loss = nn.MSELoss()
        loss = loss(output, target)

        # Backward pass to calculate the gradients
        loss.backward()

        for name, param in model.named_parameters():
            if "weight" in name:
                # Print the mean absolute gradient of the weights
                print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
    return ExampleDeepNeuralNetwork, print_gradients


@app.cell
def _(ExampleDeepNeuralNetwork, print_gradients, torch):
    layer_sizes = [3, 3, 3, 3, 3, 1]

    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_without_shortcut, sample_input)
    return layer_sizes, sample_input


@app.cell
def _(
    ExampleDeepNeuralNetwork,
    layer_sizes,
    print_gradients,
    sample_input,
    torch,
):
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""connecting attn and linear layers in transformer block""")
    return


@app.cell
def _(FeedForward, LayerNorm, MultiHeadAttention, nn):
    class TransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"],
            )
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        def forward(self, x):
            # Shortcut connection for attention block
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            # Shortcut connection for feed forward block
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            return x
    return (TransformerBlock,)


@app.cell
def _(GPT_CONFIG_124M, TransformerBlock, torch):
    torch.manual_seed(123)

    _x = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    _block = TransformerBlock(GPT_CONFIG_124M)
    _output = _block(_x)

    print("Input shape:", _x.shape)
    print("Output shape:", _output.shape)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Coding GPT model""")
    return


@app.cell
def _(LayerNorm, TransformerBlock, nn, torch):
    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )

            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits
    return (GPTModel,)


@app.cell
def _(GPTModel, GPT_CONFIG_124M, batch, out, torch):
    torch.manual_seed(123)
    gpt_model = GPTModel(GPT_CONFIG_124M)

    gpt_out = gpt_model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", gpt_out.shape)
    print(out)
    return (gpt_model,)


@app.cell
def _(gpt_model):
    gpt_total_params = sum(p.numel() for p in gpt_model.parameters())
    print(f"\nTotal number of parameters: {gpt_total_params:,}")
    return (gpt_total_params,)


@app.cell
def _(gpt_model):
    print("Token embedding layer shape:", gpt_model.tok_emb.weight.shape)
    print("Output layer shape:", gpt_model.out_head.weight.shape)
    return


@app.cell
def _(gpt_total_params, model):
    total_params_gpt2 = gpt_total_params - sum(
        p.numel() for p in model.out_head.parameters()
    )
    print(
        f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}"
    )
    return


@app.cell
def _(gpt_total_params):
    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = gpt_total_params * 4

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total size of the model: {total_size_mb:.2f} MB")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""generating text""")
    return


@app.cell
def _(torch):
    def generate_text_simple(model, idx, max_new_tokens, context_size):
        # idx is (batch, n_tokens) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = model(idx_cond)

            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx
    return (generate_text_simple,)


@app.cell
def _(tokenizer, torch):
    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    return (encoded_tensor,)


@app.cell
def _(GPT_CONFIG_124M, encoded_tensor, generate_text_simple, model):
    model.eval()  # disable dropout

    gpt2_out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Output:", gpt2_out)
    print("Output length:", len(gpt2_out[0]))
    return (gpt2_out,)


@app.cell
def _(gpt2_out, tokenizer):
    decoded_text = tokenizer.decode(gpt2_out.squeeze(0).tolist())
    print(decoded_text)
    return


if __name__ == "__main__":
    app.run()
