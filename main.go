package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Value struct {
	Data      float64
	Grad      float64
	Label     string
	Op        Operation
	Prev      []*Value
	_Backward func(out *Value)
}

type Operation int64

const (
	Nil Operation = iota
	Add
	Mul
	Neg
	Sub
	Pow
	Div
	Tanh
	Exp
)

func NewValue(data float64, label string) *Value {
	value := &Value{
		Data:      data,
		Grad:      0.0,
		Label:     label,
		Op:        Nil,
		Prev:      nil,
		_Backward: func(out *Value) {},
	}

	return value
}

func (v Value) Print() {
	fmt.Printf("Value(data=%f, grad=%f)\n", v.Data, v.Grad)
}

func (v1 Value) Add(v2 *Value) *Value {
	out := NewValue(v1.Data+v2.Data, "")
	out.Op = Add
	out.Prev = []*Value{&v1, v2}

	out._Backward = func(out *Value) {
		(&v1).Grad += 1.0 * out.Grad
		v2.Grad += 1.0 * out.Grad
	}

	return out
}

func (v1 Value) Mul(v2 *Value) *Value {
	out := NewValue(v1.Data*v2.Data, "")
	out.Op = Mul
	out.Prev = []*Value{&v1, v2}

	out._Backward = func(out *Value) {
		(&v1).Grad += v2.Data * out.Grad
		v2.Grad += v1.Data * out.Grad
	}

	return out
}

func (v1 Value) Neg() *Value {
	v2 := NewValue(-1, "")
	out := NewValue(v1.Data*v2.Data, "")
	out.Op = Neg
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) Sub(v2 *Value) *Value {
	out := v1.Add(v2.Neg())
	out.Op = Sub

	return out
}

func (v1 Value) Pow(v2 *Value) *Value {
	out := NewValue(float64(math.Pow(v1.Data, v2.Data)), "")
	out.Op = Pow
	out.Prev = []*Value{&v1, v2}

	out._Backward = func(out *Value) {
		(&v1).Grad += (v2.Data * math.Pow(v1.Data, v2.Data-1)) * out.Grad
		v2.Grad += (out.Data * math.Log(v1.Data)) * out.Grad
	}

	return out
}

func (v1 Value) Div(v2 *Value) *Value {
	out := NewValue(v1.Data*v2.Pow(NewValue(-1, "")).Data, "")
	out.Op = Div
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v Value) Tanh() *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := NewValue(t, "")
	out.Op = Tanh
	out.Prev = []*Value{&v}

	out._Backward = func(out *Value) {
		(&v).Grad += (1 - math.Pow(t, 2)) * out.Grad
	}

	return out
}

func (v Value) Exp() *Value {
	x := v.Data
	out := NewValue(math.Exp(x), "")
	out.Op = Exp
	out.Prev = []*Value{&v}

	out._Backward = func(out *Value) {
		(&v).Grad += out.Data * out.Grad
	}

	return out
}

func (v Value) Backward() {
	var topo []*Value
	var visited []*Value
	var BuildTopo func(v *Value)
	v.Grad = 1.0

	BuildTopo = func(v *Value) {
		for _, node := range visited {
			if node == v {
				return
			}
		}
		visited = append(visited, v)
		for _, child := range v.Prev {
			BuildTopo(child)
		}
		topo = append(topo, v)
	}

	BuildTopo(&v)

	for i := len(topo) - 1; i >= 0; i-- {
		topo[i]._Backward(topo[i])
		//fmt.Println(topo[i].Data, topo[i].Grad)
	}
}

func GenerateRandomFloat(min float64, max float64) float64 {
	rand.Seed(time.Now().UnixNano())

	upper := math.Ceil(min)
	lower := math.Floor(max)
	n := rand.Float64()*(upper-lower) + lower

	return n
}

type Neuron struct {
	Weights []*Value
	Bias    *Value
}

func NewNeuron(nin int) *Neuron {
	var weights []*Value
	for i := 0; i < nin; i++ {
		weights = append(weights, NewValue(GenerateRandomFloat(-1, 1), ""))
	}
	bias := NewValue(GenerateRandomFloat(-1, 1), "")

	neuron := &Neuron{
		Weights: weights,
		Bias:    bias,
	}

	return neuron
}

func (n Neuron) Call(x []float64) *Value {
	act := n.Bias

	for i, _ := range x {
		act = act.Add(n.Weights[i].Mul(NewValue(x[i], "")))
	}

	out := act.Tanh()
	return out
}

func (n Neuron) Parameters() []*Value {
	return append(n.Weights, n.Bias)
}

type Layer struct {
	Neurons []*Neuron
}

func NewLayer(nin int, nout int) *Layer {
	var neurons []*Neuron

	for i := 0; i < nout; i++ {
		neurons = append(neurons, NewNeuron(nin))
	}

	layer := &Layer{
		Neurons: neurons,
	}

	return layer
}

func (l Layer) Call(x []float64) []*Value {
	var outs []*Value

	for _, n := range l.Neurons {
		outs = append(outs, n.Call(x))
	}

	return outs
}

func (l Layer) Parameters() []*Value {
	var params []*Value

	for _, neuron := range l.Neurons {
		params = append(params, neuron.Parameters()...)
	}

	return params
}

type MLP struct {
	Layers []*Layer
}

func NewMLP(nin int, nouts []int) *MLP {
	sz := append([]int{nin}, nouts...)

	var layers []*Layer

	for i, _ := range nouts {
		layers = append(layers, NewLayer(sz[i], sz[i+1]))
	}

	mlp := &MLP{
		Layers: layers,
	}

	return mlp
}

func (mlp MLP) Call(x []float64) *Value {
	var l []*Value

	for _, layer := range mlp.Layers {
		l = layer.Call(x)
	}

	return l[0]
}

func (mlp MLP) Parameters() []*Value {
	var params []*Value

	for _, layer := range mlp.Layers {
		params = append(params, layer.Parameters()...)
	}

	return params
}

func main() {
	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	ys := []float64{1.0, -1.0, -1.0, 1.0}

	n := NewMLP(3, []int{4, 4, 1})

	for i := 0; i < 100; i++ {
		var ypred []*Value
		loss := NewValue(0.0, "")

		for _, x := range xs {
			ypred = append(ypred, n.Call(x))
		}

		for i, _ := range ypred {
			ygt := NewValue(ys[i], "")
			yout := ypred[i]

			loss = loss.Add(
				yout.Sub(ygt).Pow(
					NewValue(2, ""),
				),
			)
		}

		for _, p := range n.Parameters() {
			p.Grad = 0.0
		}

		loss.Backward()

		for _, p := range n.Parameters() {
			p.Data += -0.1 * p.Grad
		}

		fmt.Printf("Step: %d, Loss: %f\n", i, loss.Data)

		if i == 99 {
			fmt.Println("Final predictions:")
			for _, pred := range ypred {
				fmt.Println(pred)
			}
		}
	}
}
