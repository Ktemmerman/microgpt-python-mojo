# Mojo

Mojo is interesting to me because it tries to combine Python-like usability with the control and speed expected from a systems language.

## Where Mojo fits in the ML stack

The history of ML systems can be viewed as a series of layers that removed different kinds of complexity:

1. **Performance — BLAS and LAPACK.**The Basic Linear Algebra Subprograms (BLAS) and LAPACK solved the problem of Hardware Primitives. They provided standardized, highly optimized implementations of matrix operations such as general matrix multiply (GEMM). This layer ensures that C = A @ B runs at near-peak silicon speed, regardless of the language calling it. 
2. **Usability — NumPy.** NumPy solved the problem of Developer Velocity. By wrapping  low-level BLAS routines in high-level Python (Harris et al. 2020), it allowed scientists to write code in a friendly language while executing it in optimized C/Fortran. This “Vectorization” pattern, where the slow language handles logic and the fast language handles loops, became the standard contract for scientific computing. 
3. **Differentiation — Theano, TensorFlow, and PyTorch.** Deep Learning Frameworks (Theano7, TensorFlow,  PyTorch) solved the problem of Gradient Computation. While NumPy required manual derivation of backpropagation gradients (error-prone and slow), these frameworks introduced Automatic Differentiation via the computational graph. This turned the chain rule into a software primitive, allowing researchers to define forward passes and get backward passes for free.
4. **Integration — Mojo and Modular.** My interpretation is that Mojo is trying to reduce the complexity created by all these separate layers: a productive high-level language on one side, low-level kernels and hardware-specific code on the other, and many point solutions between them. Mojo's idea is to preserve the usability of python and combinding it with the **performance and safety of compiled languages such as C++ and Rust**. 

## 1. MLIR lets Mojo support multiple compilation modes

One of the interesting things about Mojo is that it supports different ways of executing the same program. Code can be compiled and run directly, or it can be compiled ahead of time into an optimized executable. Importantly, these are not separate language implementations: both are built on the same compiler foundation, based on **MLIR (Multi-Level Intermediate Representation)**.

MLIR is cool because it gradually turns high-level source code into the low-level instructions that a CPU or GPU can execute in an interesting way.

A traditional compiler often follows a relatively fixed pipeline:

```text
Source code
   ↓
High-level intermediate representation
   ↓
Lower-level intermediate representation
   ↓
Machine-oriented representation
   ↓
CPU or GPU instructions
```

At every stage, some high-level information may disappear. For example, the compiler might initially understand an operation as:

```text
matrix_multiply(A, B)
```

Later, that same operation may be represented as nested loops:

```text
for i
  for j
    for k
      load
      multiply
      add
      store
```

Once the matrix multiplication has been broken down into loops, loads, and arithmetic operations, the compiler may no longer explicitly know that the original intent was "perform a matrix multiplication." That matters because high-level information can be extremely useful for optimization.

MLIR takes a more flexible approach. Instead of forcing everything into one intermediate representation too early, it allows the compiler to represent a program at **multiple levels of abstraction**.

For example, a program might conceptually move through stages like:

```text
Mojo source
   ↓
Tensor or matrix operations
   ↓
Loops
   ↓
Memory operations
   ↓
CPU or GPU-specific operations
   ↓
Machine code
```

MLIR organizes these different representations using **dialects**. A dialect is essentially a vocabulary for describing operations at a particular level of abstraction or for a particular domain. One dialect might understand high-level operations such as:

```text
matrix_multiply A, B
```

Another might represent loops. Another might describe memory loads and stores. Yet another might contain operations designed specifically for GPUs.

The compiler can then gradually move between these representations. This process is called **lowering**:

```text
matrix multiplication
        ↓
loops
        ↓
memory operations
        ↓
CPU/GPU operations
        ↓
machine instructions
```

The advantage is that MLIR does not have to lower an operation immediately. It can keep the high-level meaning around for as long as that information is useful. For example, if the compiler still knows that an operation is a matrix multiplication and the target is a GPU, it can choose a specialized GPU implementation. If that operation had already been converted into thousands of individual loads, multiplications, and additions, recognizing the same opportunity would be much harder.

MLIR therefore gives the compiler the ability to **optimize an operation at the level where it makes the most sense, and only then lower it toward the hardware**.

What makes this especially powerful is that MLIR does not just provide different levels of representation; it also provides a shared set of tools for transforming code between them. Those transformations happen through **compiler passes**. A pass is simply a step that analyzes or rewrites part of the program: removing unnecessary operations, simplifying loops, vectorizing calculations, or converting generic operations into instructions tailored for a CPU or GPU.

```text
Program
   ↓
Remove unnecessary operations
   ↓
Simplify loops
   ↓
Vectorize calculations
   ↓
Lower to CPU/GPU-specific operations
```

Because these passes operate on MLIR's common infrastructure, many of them can be reused across different dialects and stages of compilation rather than being reinvented for every representation.

Underneath this sits another important idea: **SSA, or Static Single Assignment**. In SSA form, each intermediate value is assigned only once:

```text
x1 = 10
x2 = x1 + 5
x3 = x2 * 2
```

That may look like a small implementation detail, but it gives the compiler a much clearer picture of where every value comes from and how it flows through the program. That makes analyses and optimizations easier and safer.

Together, dialects, compiler passes, and SSA are what make MLIR more than just a collection of intermediate representations. They form a common framework in which code can remain high-level while that is useful, be optimized at the right level of abstraction, and then be progressively lowered toward the target hardware. For Mojo, that is particularly valuable: the language can expose expressive, high-level features to the programmer while the compiler still has the machinery to specialize the same program for different execution modes and hardware targets.

This repository makes that distinction visible with two commands for [`microgpt.mojo`](microgpt.mojo):

```bash
mojo run microgpt.mojo
mojo build -O3 microgpt.mojo -o build/microgpt_mojo
```

Both enter Mojo's MLIR-based pipeline. `mojo run` compiles and immediately executes the result; `mojo build` emits an optimized standalone executable.

## 2. MLIR connects high-level Mojo to native code

Another cool feature that MLIR provides, and the one that interests me most about Mojo, is optimization. Because MLIR can optimize code while useful high-level structure is still available, Mojo lets you write the model using normal functions and collection abstractions without having to manually express it in low-level, machine-oriented terms.

The linear layer is a good example:

```mojo
def linear(mut tape: Tape, x: List[Int], weights: Matrix) -> List[Int]:
    return [dot(tape, row, x, 0, len(x)) for row in weights]
```

At the source level, this is written in a very direct way: apply `dot` to each row of `weights` and collect the results in a list. The programmer does not have to manually spell out the lower-level loops, temporary values, or machine instructions needed to perform that work.

As the program moves through the compiler, however, `dot`, the list comprehension, and the surrounding loops are not treated as untouchable runtime abstractions. The compiler can inspect their structure, rewrite them, combine operations, and eventually lower them into a more efficient native representation.

Building with `-O3` tells the compiler to apply more aggressive optimization passes:

```bash
mojo build -O3 microgpt.mojo -o build/microgpt_mojo
```

Where it is safe and useful, those passes can inline function calls, simplify or restructure loops, remove work whose result is unnecessary, vectorize repeated arithmetic, and improve how data is accessed in memory. The key point is that the source can stay readable while the generated program becomes much more specialized.

Mojo can also provide the compiler with information that is known **before the program runs**:

```mojo
comptime N_EMBD = 16
comptime N_HEAD = 4
comptime HEAD_DIM = N_EMBD // N_HEAD
comptime Matrix = List[List[Int]]
```

Here, `N_EMBD` and `N_HEAD` are compile-time constants, so the compiler already knows that `HEAD_DIM` is `4`. Likewise, `Matrix` is resolved as a type alias during compilation rather than determined at runtime.

That means these are not values the program has to discover while it is executing. They become facts the compiler can use while transforming the program, eliminating work that would otherwise have to happen at runtime.

This `microGPT` implementation targets the CPU, but the same multi-level approach is designed to support specialization for other processors and accelerators without requiring a new source-level programming model.

## 3. Types make ownership behavior explicit

Another interesting part of Mojo is that it gives programmers explicit control over how data behaves: types can define how values are created, borrowed, copied, moved, and destroyed instead of leaving those decisions entirely to the runtime.

That is visible in the `matrix` function:

```mojo
def matrix(
    mut tape: Tape,
    mut rng: NormalRandom,
    rows: Int,
    cols: Int,
    std: Float32 = 0.08,
) -> Matrix:
    var result = Matrix()
    for _ in range(rows):
        var row = List[Int]()
        for _ in range(cols):
            row.append(tape.leaf_node(Float64(rng.step_normal(stddev=std)[0])))
        result.append(row^)
    return result^
```

A few small pieces of syntax tell us how values move through the function.

`mut tape` and `mut rng` mean that the function receives mutable borrows. It is allowed to change those objects, but it does not take ownership of them. When the function returns, the caller still owns the same `Tape` and random-number generator.

The `^` operator means something different: it transfers ownership of a value. Once a row has been constructed, `row^` moves that row into `result` rather than creating another copy of it:

```mojo
result.append(row^)
```

The same thing happens at the end of the function:

```mojo
return result^
```

The completed matrix is transferred back to the caller.

This matters because large data structures can be expensive to duplicate. If the program only needs to hand a value from one place to another, moving it is often preferable to copying all of its contents.

When the model actually needs two independent values, the copy is explicit:

```mojo
var residual = x.copy()
```

Here the intent is different: `residual` must remain separate from `x`, so a real copy is requested.

That makes an important performance distinction visible directly in the source:

```text
borrow  → use an existing value without taking ownership
move    → transfer an existing value without duplicating it
copy    → create a separate independent value
```

For performance-sensitive code, that explicitness is valuable.

## 4. A choice for elegance and simplicity

This one is less about performance and more about aesthetics: I like language features that make code more elegant and simpler.

Constructors are necessary, but writing them out field by field is often repetitive and just ugly. They add clutter without telling you much that the struct definition does not already say. Mojo's `@fieldwise_init` decorator removes that boilerplate while keeping the actual structure of the type completely visible:

```mojo
@fieldwise_init
struct Value(ImplicitlyCopyable):
    var data: Float64
    var grad: Float64
    var child0: Int
    var child1: Int
    var local_grad0: Float64
    var local_grad1: Float64
```

The fields, their names, and their types are still right there in the source. The decorator simply generates the mechanical constructor that would otherwise have to be written by hand.

The generated constructor can then be used normally throughout the autograd tape:

```mojo
self.values.append(
    Value(data, 0.0, child0, child1, local_grad0, local_grad1)
)
```

It is a small design choice, but it reflects something larger: the designers seem to value elegance, simplicity, and the experience of the programmer. There is a bit of the Ruby philosophy in that idea—the language should not merely satisfy the machine; it should make programming pleasant for the person writing the code.

## Why this matters

Taken together, these choices suggest a language that cares deeply about raw performance. About pushing hardware as close to its physical limits as possible, while still valuing elegance, simplicity, and the experience of the person writing the program. That balance is what makes the language interesting to me.

