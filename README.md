# micrograd-go

A Go implementation of Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) Autograd engine. Like the original Python version, this implements backpropagation over a DAG of Value operations with a PyTorch-like API (though I didn't refer to the Go bindings).

### Installation

```bash
go get -v github.com/farrelmahaztra/micrograd-go
```

### Usage

The following is an example of how you can use micrograd-go, taken from one of the examples in the Python guide.

```go
package main

import (
	"github.com/farrelmahaztra/micrograd-go/pkg/engine"
)

func main() {
	// Inputs x1 and x2
	x1 := engine.NewValue(2.0)
	x2 := engine.NewValue(0.0)

	// Weights w1 and w2
	w1 := engine.NewValue(-3.0)
	w2 := engine.NewValue(1.0)

	// Bias of the neuron
	b := engine.NewValue(6.8813735870195432)

	// x1*w1 + x2*w2 + b
	x1w1 := x1.Mul(w1)
	x2w2 := x2.Mul(w2)

	x1w1x2w2 := x1w1.Add(x2w2)

	n := x1w1x2w2.Add(b)

	// tanh(x1*w1 + x2*w2 + b)
	o := n.Tanh()

	// Calculate gradients
	o.Backward()
}
```

Or alternatively, try the full training loop by doing:

```bash
go run cmd/micrograd-go/main.go
```