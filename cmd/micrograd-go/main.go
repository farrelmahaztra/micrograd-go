package main

import (
	"fmt"

	"github.com/farrelmahaztra/micrograd-go/pkg/engine"
	"github.com/farrelmahaztra/micrograd-go/pkg/nn"
)

func main() {
	// Training loop
	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	ys := []float64{1.0, -1.0, -1.0, 1.0}

	n := nn.NewMLP(3, []int{4, 4, 1})

	for i := 1; i <= 10000; i++ {
		var ypred []*engine.Value
		loss := engine.NewValue(0.0, "Initial loss value")

		for _, x := range xs {
			ypred = append(ypred, n.Call(x))
		}

		for i, _ := range ypred {
			ygt := engine.NewValue(ys[i], "ygt")
			yout := ypred[i]

			loss = loss.Add(
				(yout.Sub(ygt, "yout-ygt")).Pow(2, "Pow for mse"), "Add to loss")
		}

		for _, p := range n.Parameters() {
			p.Grad = 0.0
		}

		loss.Backward()

		for _, p := range n.Parameters() {
			lr := -0.1
			p.Data += p.Grad * lr
		}

		fmt.Printf("Step: %d, Loss: %f\n", i, loss.Data)

		if i == 10000 {
			fmt.Println("Final predictions:")
			for _, pred := range ypred {
				pred.Print()
			}
		}
	}
}
