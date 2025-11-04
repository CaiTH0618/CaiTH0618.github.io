---
layout: page
title: Backpropagation
---

Update: 2025-11-04

## The Basic Idea of Training

During neural network training, we hope to iteratively update the parameters/weights of an NN so that it can gradually turn into a function that has the expected mapping between input and output. 

In order to update the parameters of an NN, we have to compute the gradient of the loss $l$ with respect to a parameter $\theta$, which is 

$$
\frac{\partial l}{\partial \theta}
$$

Having the gradients, we can update all the parameters through an update function:

$$
\theta' = \text{UPDATE}(\theta, \frac{\partial l}{\partial \theta})
$$

## The Math Behind It

Say $f$ is an operator in an NN, expressed as

$$
y = f(x)
$$

where $x$ and $y$ are general representations for the input tensors and output tensors of this operator (for easier reading). Given an input to the NN, during forward propagation, we obtain the values of $x$ (derived from the computation of previous operators) and $y$ (through the computation of $f$). We also obtain the value of the loss $l$ after the loss computation at the end. 

Assuming that we already have the gradients for $y$, we can compute the gradients for $x$ using the chain rule:

$$
\frac{\partial l}{\partial x} = \frac{\partial l}{\partial y} \cdot \frac{\partial y}{\partial x}
$$

This means that we can compute the gradient for every parameter of the NN from the output layer back to the input layer (from tail to head), hence the name "backpropagation".

## How AI Frameworks Implement It

Mathematically, we need a Jacobian matrix composed of all the partial derivatives $\partial y_j / \partial x_i$. However, AI frameworks don't compute full Jacobian matrices because it would be computationally expensive and memory-intensive. Instead, they rely on manual implementation of a backward function for each operator that efficiently computes only the needed gradients.

Take PyTorch as an example. It requires developers to write a backward function, which takes as input the gradients of the output, and outputs the gradients of the input.

```python
# WARNING: AI generated code
class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        # Save inputs for backward pass
        ctx.save_for_backward(a, b)
        return a * b
    
    @staticmethod
    def backward(ctx, grad_out):
        # Retrieve saved inputs
        a, b = ctx.saved_tensors
        # Apply chain rule
        return grad_out * b, grad_out * a
```

In TensorFlow, custom gradients are defined using the `@tf.custom_gradient` decorator:

```python
# WARNING: AI generated code
@tf.custom_gradient
def mul_op(a, b):
    # Forward pass
    result = a * b
    
    def grad(grad_out):
        # Backward pass: apply chain rule
        return grad_out * b, grad_out * a
    
    return result, grad
```

In JAX, custom gradients are implemented using `jax.custom_vjp` (vector-Jacobian product):

```python
# WARNING: AI generated code
@jax.custom_vjp
def mul_op(a, b):
    # Forward pass
    return a * b

def mul_fwd(a, b):
    # Forward pass with saved values for backward
    return a * b, (a, b)

def mul_bwd(residuals, grad_out):
    # Backward pass
    a, b = residuals
    return (grad_out * b, grad_out * a)

# Register the forward and backward functions
mul_op.defvjp(mul_fwd, mul_bwd)
```

All three frameworks follow the same principle: they avoid computing full Jacobian matrices by implementing efficient backward functions that directly compute the required gradients using the chain rule.

However, frameworks will automatically compute gradients for operators that are implemented by composing the framework's differentiable primitives (for example, `add`, `mul`, `sin`). The framework records the computation (graph or trace) during the forward pass and applies the chain rule to produce the backward pass.

The automatic gradients only work when the implementation uses the framework's differentiable ops and is visible to its tracer/tape. If you use raw non‑differentiable code (e.g., plain NumPy), integer-only operations, destructive in‑place updates, or custom low‑level kernels/primitives, you may need to register or implement an explicit backward/grad/VJP for correct gradients. Higher‑order derivatives and some advanced control‑flow patterns can also require extra care.
