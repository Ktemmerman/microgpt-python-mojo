from std.collections import Set
from std.io import FileHandle
from std.math import exp, log, sqrt
from std.pathlib import Path
from std.random import seed, shuffle
from std.random.philox import NormalRandom, Random
from std.subprocess import run

comptime N_LAYER = 1
comptime N_EMBD = 16
comptime BLOCK_SIZE = 16
comptime N_HEAD = 4
comptime HEAD_DIM = N_EMBD // N_HEAD
comptime LEARNING_RATE: Float64 = 0.01
comptime BETA1: Float64 = 0.85
comptime BETA2: Float64 = 0.99
comptime ADAM_EPS: Float64 = 1e-8
comptime NUM_STEPS = 1000
comptime Matrix = List[List[Int]]
comptime KVCache = List[Matrix]

@fieldwise_init
struct Value(ImplicitlyCopyable):
    var data: Float64
    var grad: Float64
    var child0: Int
    var child1: Int
    var local_grad0: Float64
    var local_grad1: Float64

@fieldwise_init
struct Tape:
    var values: List[Value]

    def leaf_node(mut self, data: Float64) -> Int: return self.node(data)
    def node(mut self, data: Float64, child0: Int = -1, child1: Int = -1, local_grad0: Float64 = 0.0, local_grad1: Float64 = 0.0) -> Int: self.values.append(Value(data, 0.0, child0, child1, local_grad0, local_grad1)); return len(self.values) - 1
    def add(mut self, a: Int, b: Int) -> Int: return self.node(self.values[a].data + self.values[b].data, a, b, 1.0, 1.0)
    def add(mut self, a: Int, b: Float64) -> Int: return self.node(self.values[a].data + b, a, local_grad0=1.0)
    def mul(mut self, a: Int, b: Int) -> Int: return self.node(self.values[a].data * self.values[b].data, a, b, self.values[b].data, self.values[a].data)
    def mul(mut self, a: Int, b: Float64) -> Int: return self.node(self.values[a].data * b, a, local_grad0=b)
    def power(mut self, a: Int, exponent: Float64) -> Int: var data = self.values[a].data; return self.node(data**exponent, a, local_grad0=exponent * data ** (exponent - 1.0))
    def log(mut self, a: Int) -> Int: var data = self.values[a].data; return self.node(log(data), a, local_grad0=1.0 / data)
    def exp(mut self, a: Int) -> Int: var result = exp(self.values[a].data); return self.node(result, a, local_grad0=result)
    def relu(mut self, a: Int) -> Int: var data = self.values[a].data; return self.node(data if data > 0.0 else 0.0, a, local_grad0=1.0 if data > 0.0 else 0.0)

    def backward(mut self, loss: Int):
        self.values[loss].grad = 1.0
        var i = loss + 1
        while i > 0:
            i -= 1
            var value = self.values[i]
            if value.child0 >= 0: self.values[value.child0].grad += value.local_grad0 * value.grad
            if value.child1 >= 0: self.values[value.child1].grad += value.local_grad1 * value.grad

    def reset_to(mut self, size: Int):
        while len(self.values) > size: _ = self.values.pop()

@fieldwise_init
struct Layer:
    var attn_wq: Matrix
    var attn_wk: Matrix
    var attn_wv: Matrix
    var attn_wo: Matrix
    var mlp_fc1: Matrix
    var mlp_fc2: Matrix

@fieldwise_init
struct Model:
    var wte: Matrix
    var wpe: Matrix
    var lm_head: Matrix
    var layers: List[Layer]

def matrix(mut tape: Tape, mut rng: NormalRandom, rows: Int, cols: Int, std: Float32 = 0.08) -> Matrix:
    var result = Matrix()
    for _ in range(rows):
        var row = List[Int]()
        for _ in range(cols): row.append(tape.leaf_node(Float64(rng.step_normal(stddev=std)[0])))
        result.append(row^)
    return result^

def dot(mut tape: Tape, a: List[Int], b: List[Int], start: Int, size: Int) -> Int:
    var total = tape.leaf_node(0.0)
    for i in range(start, start + size): total = tape.add(total, tape.mul(a[i], b[i]))
    return total

def weighted_sum(mut tape: Tape, weights: List[Int], values: Matrix, column: Int) -> Int:
    var total = tape.leaf_node(0.0)
    for i in range(len(weights)): total = tape.add(total, tape.mul(weights[i], values[i][column]))
    return total

def linear(mut tape: Tape, x: List[Int], weights: Matrix) -> List[Int]:
    return [dot(tape, row, x, 0, len(x)) for row in weights]

def softmax(mut tape: Tape, logits: List[Int]) -> List[Int]:
    var maximum = tape.values[logits[0]].data
    for value in logits:
        if tape.values[value].data > maximum: maximum = tape.values[value].data
    var exps = [tape.exp(tape.add(value, -maximum)) for value in logits]
    var total = tape.leaf_node(0.0)
    for item in exps: total = tape.add(total, item)
    var inverse_total = tape.power(total, -1.0)
    return [tape.mul(item, inverse_total) for item in exps]

def rmsnorm(mut tape: Tape, x: List[Int]) -> List[Int]:
    var mean_square = tape.leaf_node(0.0)
    for value in x: mean_square = tape.add(mean_square, tape.mul(value, value))
    mean_square = tape.mul(mean_square, 1.0 / Float64(len(x)))
    var scale = tape.power(tape.add(mean_square, 1e-5), -0.5)
    return [tape.mul(value, scale) for value in x]

def gpt(mut tape: Tape, model: Model, token_id: Int, pos_id: Int, mut keys: KVCache, mut values: KVCache) -> List[Int]:
    var x = [tape.add(model.wte[token_id][i], model.wpe[pos_id][i]) for i in range(N_EMBD)]
    x = rmsnorm(tape, x)
    for layer_index in range(N_LAYER):
        var residual = x.copy()
        x = rmsnorm(tape, x)
        var q = linear(tape, x, model.layers[layer_index].attn_wq)
        var k = linear(tape, x, model.layers[layer_index].attn_wk)
        var v = linear(tape, x, model.layers[layer_index].attn_wv)
        keys[layer_index].append(k^)
        values[layer_index].append(v^)

        var attended = List[Int]()
        for head in range(N_HEAD):
            var start = head * HEAD_DIM
            var scores = [tape.mul(dot(tape, q, keys[layer_index][t], start, HEAD_DIM), 1.0 / sqrt(Float64(HEAD_DIM))) for t in range(len(keys[layer_index]))]
            var weights = softmax(tape, scores)
            for j in range(HEAD_DIM): attended.append(weighted_sum(tape, weights, values[layer_index], start + j))
        var projected = linear(tape, attended, model.layers[layer_index].attn_wo)
        x = [tape.add(projected[i], residual[i]) for i in range(N_EMBD)]

        residual = x.copy()
        x = rmsnorm(tape, x)
        var hidden = linear(tape, x, model.layers[layer_index].mlp_fc1)
        var activated = [tape.relu(value) for value in hidden]
        projected = linear(tape, activated, model.layers[layer_index].mlp_fc2)
        x = [tape.add(projected[i], residual[i]) for i in range(N_EMBD)]
    return linear(tape, x, model.lm_head)

def adam_step(mut tape: Tape, mut m: List[Float64], mut v: List[Float64], step: Int, num_steps: Int):
    var learning_rate = LEARNING_RATE * (1.0 - Float64(step) / Float64(num_steps))
    var correction1 = 1.0 - BETA1 ** Float64(step + 1)
    var correction2 = 1.0 - BETA2 ** Float64(step + 1)
    for i in range(len(m)):
        var grad = tape.values[i].grad
        m[i] = BETA1 * m[i] + (1.0 - BETA1) * grad
        v[i] = BETA2 * v[i] + (1.0 - BETA2) * grad * grad
        var m_hat = m[i] / correction1
        var v_hat = v[i] / correction2
        tape.values[i].data -= learning_rate * m_hat / (sqrt(v_hat) + ADAM_EPS)
        tape.values[i].grad = 0.0

def main() raises:
    seed(42)
    if not Path("input.txt").exists():
        _ = run("curl -fsSL https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt -o input.txt")
    var docs = FileHandle("input.txt", "r").read().splitlines()
    shuffle(docs)
    print("num docs:", len(docs))

    var unique = Set[String]()
    for doc in docs:
        for c in doc.codepoint_slices(): unique.add(String(c))
    var uchars = List(unique)
    sort(uchars[:])
    var BOS = len(uchars)
    var vocab_size = BOS + 1
    print("vocab size:", vocab_size)

    var tape = Tape([])
    var rng = NormalRandom(seed=42)
    var wte = matrix(tape, rng, vocab_size, N_EMBD)
    var wpe = matrix(tape, rng, BLOCK_SIZE, N_EMBD)
    var lm_head = matrix(tape, rng, vocab_size, N_EMBD)
    var layers = List[Layer]()
    for _ in range(N_LAYER):
        layers.append(Layer(
            matrix(tape, rng, N_EMBD, N_EMBD), matrix(tape, rng, N_EMBD, N_EMBD),
            matrix(tape, rng, N_EMBD, N_EMBD), matrix(tape, rng, N_EMBD, N_EMBD),
            matrix(tape, rng, 4 * N_EMBD, N_EMBD), matrix(tape, rng, N_EMBD, 4 * N_EMBD),
        ))
    var model = Model(wte^, wpe^, lm_head^, layers^)
    var num_params = len(tape.values)
    print("num params:", num_params)
    var m: List[Float64] = [0.0 for _ in range(num_params)]
    var v: List[Float64] = [0.0 for _ in range(num_params)]

    for step in range(NUM_STEPS):
        var tokens: List[Int] = [BOS]
        for c in docs[step % len(docs)].codepoint_slices(): tokens.append(uchars.index(String(c)))
        tokens.append(BOS)
        var n = min(BLOCK_SIZE, len(tokens) - 1)

        var keys: KVCache = [Matrix() for _ in range(N_LAYER)]
        var values: KVCache = [Matrix() for _ in range(N_LAYER)]
        var loss = tape.leaf_node(0.0)
        for pos_id in range(n):
            var probs = softmax(tape, gpt(tape, model, tokens[pos_id], pos_id, keys, values))
            var loss_t = tape.mul(tape.log(probs[tokens[pos_id + 1]]), -1.0)
            loss = tape.add(loss, loss_t)
        loss = tape.mul(loss, 1.0 / Float64(n))

        tape.backward(loss)
        var loss_value = tape.values[loss].data
        adam_step(tape, m, v, step, NUM_STEPS)
        tape.reset_to(num_params)
        print("step", step + 1, "/", NUM_STEPS, "| loss", loss_value, end="\r")

    var temperature = 0.5
    var sample_rng = Random(seed=42)
    print("\n--- inference (new, hallucinated names) ---")
    for sample_index in range(20):
        var keys: KVCache = [Matrix() for _ in range(N_LAYER)]
        var values: KVCache = [Matrix() for _ in range(N_LAYER)]
        var token_id = BOS
        var sample = String()
        for pos_id in range(BLOCK_SIZE):
            var logits = gpt(tape, model, token_id, pos_id, keys, values)
            var probs = softmax(tape, [tape.mul(logit, 1.0 / temperature) for logit in logits])
            var draw = Float64(sample_rng.step_uniform()[0])
            var cumulative = 0.0
            token_id = BOS
            for i in range(vocab_size):
                cumulative += tape.values[probs[i]].data
                if draw <= cumulative:
                    token_id = i
                    break
            if token_id == BOS: break
            sample += uchars[token_id]
        tape.reset_to(num_params)
        print("sample", sample_index + 1, ":", sample)
