"""Chapter 3 (practical): the simplest possible language model, trained "by hand".

The model: a character BIGRAM model. It looks at ONE character and predicts the next one.
The whole model is a single table of numbers W:
    row    = the current character
    column = a score for each possible next character

Training is exactly the loop from the theory chapter:
    predict -> loss -> gradient -> update -> repeat
and the update is literally:   parameter = parameter - learning_rate * gradient

Run:
    python ch3_first_training.py                      # built-in C sample
    python ch3_first_training.py path/to/file.c       # train on a real file from data/raw
"""
import math
import sys
import time

import torch
import torch.nn.functional as F

torch.manual_seed(1337)

# ---------------------------------------------------------------- settings
LEARNING_RATE = 10.0   # size of each step downhill (try 0.1, 3, 30, 300 and watch the loss)
STEPS = 1000           # how many times we repeat the loop
BATCH = 1024           # how many (char -> next char) examples per step
PRINT_EVERY = 100
MAX_CHARS = 1_000_000  # read at most this much text from a file

SAMPLE = r"""#include <stdio.h>
#include <stdlib.h>

int add(int a, int b) {
    return a + b;
}

int main(void) {
    int x = 5;
    int y = 10;
    int z = add(x, y);
    printf("sum = %d\n", z);
    for (int i = 0; i < 3; i++) {
        printf("i = %d\n", i);
    }
    return 0;
}
"""

# ---------------------------------------------------------------- data
if len(sys.argv) > 1:
    with open(sys.argv[1], encoding="utf-8", errors="ignore") as f:
        text = f.read(MAX_CHARS)
    print(f"training on file: {sys.argv[1]}")
else:
    text = SAMPLE * 20
    print("training on the built-in C sample")

chars = sorted(set(text))
V = len(chars)  # vocabulary size = number of distinct characters
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in text])
x = data[:-1]  # current characters
y = data[1:]   # the characters that really came next
print(f"{len(text):,} characters, {V} distinct characters")
print(f"random-guess loss = ln({V}) = {math.log(V):.3f}   (the loss must start near this)\n")

# ---------------------------------------------------------------- the model
W = (torch.randn(V, V) * 0.01).requires_grad_()  # the "parameters": V x V numbers


def generate(n=200, seed_char="\n"):
    """Sample text from the model: pick the next char using the softmax probabilities."""
    ch = stoi.get(seed_char, 0)
    out = []
    for _ in range(n):
        probs = F.softmax(W[ch], dim=0)
        ch = torch.multinomial(probs, 1).item()
        out.append(itos[ch])
    return "".join(out)


print("--- text BEFORE training (random numbers) ---")
print(generate(150))

# ---------------------------------------------------------------- training loop
print("\n--- training ---")
start = time.time()
for step in range(STEPS + 1):
    ix = torch.randint(0, len(x), (BATCH,))   # pick random examples
    logits = W[x[ix]]                          # 1) predict: scores for the next char
    loss = F.cross_entropy(logits, y[ix])      # 2) loss  = -ln(probability of the right char)

    W.grad = None
    loss.backward()                            # 3) gradient of the loss w.r.t. every number in W

    with torch.no_grad():
        W -= LEARNING_RATE * W.grad            # 4) update: step downhill

    if step % PRINT_EVERY == 0:
        print(f"step {step:4d}   loss {loss.item():.3f}")

elapsed = time.time() - start
print(f"\n{STEPS} steps in {elapsed:.2f}s  ({STEPS / elapsed:.0f} steps/s on this CPU)")

print("\n--- text AFTER training ---")
print(generate(300))

# ---------------------------------------------------------------- what to try next
print("\nTry this:")
print("  1) Change LEARNING_RATE to 0.1, then 300. What happens to the loss?")
print("  2) Train on a real file:  python ch3_first_training.py data/raw/c/redis__redis/src/server.c")
print("  3) Why does the loss stop falling around a certain value? (hint: the model sees only ONE char)")
