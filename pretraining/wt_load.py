import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    from importlib.metadata import version

    pkgs = [
        "torch",
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    return


@app.cell
def _(device):
    import gc
    import time
    import torch

    def start_memory_tracking():
        """Initialize GPU memory tracking."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        else:
            print("This notebook is intended for CUDA GPUs but CUDA is not available.")

    def print_memory_usage():
        max_gpu_memory = torch.cuda.max_memory_allocated() / (
            1024**3
        )  # Convert bytes to GB
        print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")

    def cleanup():
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(3)  # some buffer time to allow memory to clear
        torch.cuda.reset_peak_memory_stats()
        max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")

    return cleanup, print_memory_usage, start_memory_tracking, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""- This code sets up the model itself""")
    return


@app.cell
def _():
    from gpt_model import GPTModel
    # If the `previous_chapters.py` file is not available locally,
    # you can import it from the `llms-from-scratch` PyPI package.
    # For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
    # E.g.,
    # from llms_from_scratch.ch04 import GPTModel

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-xl (1558M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    return BASE_CONFIG, GPTModel


@app.cell
def _(BASE_CONFIG, GPTModel, print_memory_usage, start_memory_tracking, torch):
    start_memory_tracking()

    model = GPTModel(BASE_CONFIG)
    device = torch.device("cuda")
    model.to(device)

    print_memory_usage()
    return device, model


@app.cell
def _(device, model, torch):
    # Test if the model works (no need to track memory here)
    test_input = torch.tensor([[1, 2, 3]]).to(device)
    model.eval()

    with torch.no_grad():
        model(test_input)
    return (test_input,)


@app.cell
def _(model, torch):
    # Training code would go here...

    model.train()
    torch.save(model.state_dict(), "model.pth")
    return


@app.cell
def _(cleanup, model, test_input):
    del model, test_input
    cleanup()
    return


@app.cell
def _(
    BASE_CONFIG,
    GPTModel,
    device,
    print_memory_usage,
    start_memory_tracking,
    torch,
):
    # Then load pretrained weights
    start_memory_tracking()
    model_1 = GPTModel(BASE_CONFIG)
    model_1.to(device)
    model_1.load_state_dict(
        torch.load("model.pth", map_location=device, weights_only=True)
    )
    model_1.to(device)
    model_1.eval()
    print_memory_usage()
    return (model_1,)


@app.cell
def _(cleanup, device, model_1, torch):
    # Test if the model works (no need to track memory here)
    test_input_1 = torch.tensor([[1, 2, 3]]).to(device)
    model_1.eval()
    with torch.no_grad():
        model_1(test_input_1)
    del model_1, test_input_1
    cleanup()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Loading weights sequentially
    """
    )
    return


@app.cell
def _(
    BASE_CONFIG,
    GPTModel,
    device,
    print_memory_usage,
    start_memory_tracking,
    torch,
):
    start_memory_tracking()
    model_2 = GPTModel(BASE_CONFIG).to(device)
    state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)
    print_memory_usage()
    with torch.no_grad():
        for _name, param in model_2.named_parameters():
            if _name in state_dict:
                param.copy_(state_dict[_name].to(device))
            else:
                print(f"Warning: {_name} not found in state_dict.")
    print_memory_usage()
    return model_2, param, state_dict


@app.cell
def _(cleanup, device, model_2, param, state_dict, torch):
    # Test if the model works (no need to track memory here)
    test_input_2 = torch.tensor([[1, 2, 3]]).to(device)
    model_2.eval()
    with torch.no_grad():
        model_2(test_input_2)
    del model_2, test_input_2, state_dict, param
    cleanup()
    return


@app.cell
def _(time):
    import os
    import psutil
    from threading import Thread

    def memory_usage_in_gb(func, *args, **kwargs):
        process = psutil.Process(os.getpid())

        # Measure the baseline memory usage before running the function
        baseline_mem = process.memory_info().rss / 1024**3  # in GB

        # Start monitoring memory in a separate thread
        mem_usage = []
        done = False

        def monitor_memory():
            while not done:
                mem_usage.append(process.memory_info().rss / 1024**3)  # Convert to GB
                time.sleep(0.1)

        t = Thread(target=monitor_memory)
        t.start()

        # Run the function
        func(*args, **kwargs)

        # Stop monitoring
        done = True
        t.join()

        peak_mem_usage_gb = max(mem_usage) - baseline_mem
        return peak_mem_usage_gb

    return memory_usage_in_gb, os


@app.cell
def _(
    BASE_CONFIG,
    GPTModel,
    device,
    memory_usage_in_gb,
    print_memory_usage,
    start_memory_tracking,
    torch,
):
    def load_sequentially():
        start_memory_tracking()
        model = GPTModel(BASE_CONFIG).to(device)
        state_dict = torch.load("model.pth", map_location="cpu", weights_only=True)
        print_memory_usage()
        with torch.no_grad():
            for _name, param in model.named_parameters():
                if _name in state_dict:
                    param.copy_(state_dict[_name].to(device))
                else:  # Sequentially copy weights to the model's parameters
                    print(f"Warning: {_name} not found in state_dict.")
        print_memory_usage()

    _peak_memory_used = memory_usage_in_gb(load_sequentially)
    print(f"-> Maximum CPU memory allocated: {_peak_memory_used:.1f} GB")
    return


@app.cell
def _(
    BASE_CONFIG,
    GPTModel,
    device,
    memory_usage_in_gb,
    print_memory_usage,
    start_memory_tracking,
    torch,
):
    def load_sequentially_with_meta():
        start_memory_tracking()
        with torch.device("meta"):
            model = GPTModel(BASE_CONFIG)
        model = model.to_empty(device=device)
        state_dict = torch.load("model.pth", map_location=device, weights_only=True)
        print_memory_usage()
        with torch.no_grad():
            for _name, param in model.named_parameters():
                if _name in state_dict:
                    param.copy_(state_dict[_name])
                else:
                    print(
                        f"Warning: {_name} not found in state_dict."
                    )  # Sequentially copy weights to the model's parameters
        print_memory_usage()

    _peak_memory_used = memory_usage_in_gb(load_sequentially_with_meta)
    print(f"-> Maximum CPU memory allocated: {_peak_memory_used:.1f} GB")
    return


@app.cell
def _(
    BASE_CONFIG,
    GPTModel,
    device,
    memory_usage_in_gb,
    print_memory_usage,
    start_memory_tracking,
    torch,
):
    def baseline():
        start_memory_tracking()
        model = GPTModel(BASE_CONFIG)
        model.to(device)
        model.load_state_dict(
            torch.load("model.pth", map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()
        print_memory_usage()

    _peak_memory_used = memory_usage_in_gb(baseline)
    print(f"-> Maximum CPU memory allocated: {_peak_memory_used:.1f} GB")
    return


@app.cell
def _(
    BASE_CONFIG,
    GPTModel,
    device,
    memory_usage_in_gb,
    print_memory_usage,
    torch,
):
    def best_practices():
        with torch.device("meta"):
            model = GPTModel(BASE_CONFIG)
        model.load_state_dict(
            torch.load("model.pth", map_location=device, weights_only=True, mmap=True),
            assign=True,
        )
        print_memory_usage()

    _peak_memory_used = memory_usage_in_gb(best_practices)
    print(f"-> Maximum CPU memory allocated: {_peak_memory_used:.1f} GB")
    return


@app.cell
def _(BASE_CONFIG, GPTModel, os, torch):
    model_3 = GPTModel(BASE_CONFIG)
    state_dict_1 = model_3.state_dict()
    os.makedirs("model_parameters", exist_ok=True)
    for _name, param_1 in state_dict_1.items():
        torch.save(param_1.cpu(), f"model_parameters/{_name}.pt")
    del model_3
    return


@app.cell
def _(
    BASE_CONFIG,
    GPTModel,
    device,
    memory_usage_in_gb,
    os,
    print_memory_usage,
    start_memory_tracking,
    torch,
):
    def load_individual_weights():
        start_memory_tracking()
        with torch.device("meta"):
            model = GPTModel(BASE_CONFIG)
        model = model.to_empty(device=device)
        print_memory_usage()
        param_dir = "model_parameters"
        with torch.no_grad():
            for _name, param in model.named_parameters():
                weight_path = os.path.join(param_dir, f"{_name}.pt")
                if os.path.exists(weight_path):
                    param_data = torch.load(
                        weight_path, map_location="cpu", weights_only=True
                    )
                    param.copy_(param_data)
                    del param_data
                else:
                    print(f"Warning: {_name} not found in {param_dir}.")
        print_memory_usage()

    _peak_memory_used = memory_usage_in_gb(load_individual_weights)
    print(f"-> Maximum CPU memory allocated: {_peak_memory_used:.1f} GB")  # Free memory
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
