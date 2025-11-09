import sys  # noqa: E402
from pathlib import Path  # noqa: E402

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import setup_project_imports  # noqa: E402

setup_project_imports()  # noqa: E402

import tiktoken  # noqa: E402
import torch  # noqa: E402
import chainlit  # noqa: E402

from gpt_model import GPTModel  # noqa: E402
from gpt_model import (  # noqa: E402
    text_to_token_ids,
    token_ids_to_text,
)

from utils.train import generate  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with pretrained weights from OpenAI.
    The code is similar to chapter 5.
    The model will be downloaded automatically if it doesn't exist in the current folder, yet.
    """

    CHOOSE_MODEL = "gpt2-small (124M)"  # Optionally replace with another model from the model_configs dir below

    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # Map to HF model name
    model_name_map = {
        "124M": "gpt2",
        "355M": "gpt2-medium",
        "774M": "gpt2-large",
        "1558M": "gpt2-xl",
    }
    model_name = model_name_map[model_size]

    gpt = GPTModel.from_pretrained(model_name)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")

    return tokenizer, gpt, BASE_CONFIG


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=50,
        context_size=model_config["context_length"],
        top_k=1,
        temperature=0.0
    )

    text = token_ids_to_text(token_ids, tokenizer)

    await chainlit.Message(
        content=f"{text}",  # This returns the model response to the interface
    ).send()