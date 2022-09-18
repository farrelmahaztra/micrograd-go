package main

import (
	"fmt"
	"math"
	"math/rand"
	"os/exec"
	"strings"
	"time"
)

type Value struct {
	Id        string
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

func (op Operation) String() string {
	switch op {
	case Nil:
		return "No operation"
	case Add:
		return "Add"
	case Mul:
		return "Mul"
	case Neg:
		return "Neg"
	case Sub:
		return "Sub"
	case Pow:
		return "Pow"
	case Div:
		return "Div"
	case Tanh:
		return "Tanh"
	case Exp:
		return "Exp"
	}

	return "N/A"
}

func NewValue(data float64, label string) *Value {
	uuid, _ := exec.Command("uuidgen").Output()
	value := &Value{
		Id:        string(uuid),
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
	fmt.Printf("( data=%f, grad=%f, op=%s, ", v.Data, v.Grad, v.Op.String())
	fmt.Println("prev=", v.Prev, ")")
}

func (v1 Value) Add(v2 *Value, label string) *Value {
	out := NewValue(v1.Data+v2.Data, label)
	out.Op = Add
	out.Prev = []*Value{&v1, v2}

	out._Backward = func(out *Value) {
		(&v1).Grad += 1.0 * out.Grad
		v2.Grad += 1.0 * out.Grad
	}

	return out
}

func (v1 Value) Mul(v2 *Value, label string) *Value {
	out := NewValue(v1.Data*v2.Data, label)
	out.Op = Mul
	out.Prev = []*Value{&v1, v2}

	out._Backward = func(out *Value) {
		(&v1).Grad += v2.Data * out.Grad
		v2.Grad += v1.Data * out.Grad
	}

	return out
}

func (v1 Value) Neg(label string) *Value {
	v2 := NewValue(-1, "Negate")
	out := NewValue(v1.Data*v2.Data, label)
	out.Op = Neg
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) Sub(v2 *Value, label string) *Value {
	out := v1.Add(v2.Neg("Negate for sub"), "Add for sub")
	out.Op = Sub
	out.Label = label

	return out
}

func (v1 Value) Pow(v2 float64, label string) *Value {
	out := NewValue(float64(math.Pow(v1.Data, v2)), label)
	out.Op = Pow
	out.Prev = []*Value{&v1}

	out._Backward = func(out *Value) {
		(&v1).Grad += (v2 * math.Pow(v1.Data, v2-1)) * out.Grad
	}

	return out
}

func (v1 Value) Div(v2 *Value, label string) *Value {
	out := NewValue(v1.Data*v2.Pow(-1, "Pow for div").Data, label)
	out.Op = Div
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v Value) Tanh(label string) *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := NewValue(t, label)
	out.Op = Tanh
	out.Prev = []*Value{&v}

	out._Backward = func(out *Value) {
		(&v).Grad += (1 - math.Pow(t, 2)) * out.Grad
	}

	return out
}

func (v Value) Exp(label string) *Value {
	x := v.Data
	out := NewValue(math.Exp(x), label)
	out.Op = Exp
	out.Prev = []*Value{&v}

	out._Backward = func(out *Value) {
		(&v).Grad += out.Data * out.Grad
	}

	return out
}

func Backward(v *Value, n *MLP) {
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

	BuildTopo(v)

	for i := len(topo) - 1; i >= 0; i-- {
		topo[i]._Backward(topo[i])
	}

	// TODO: Fix this hack
	nParameters := n.Parameters()
	for _, nParam := range nParameters {
		for _, node := range topo {
			if strings.Compare(nParam.Id, node.Id) == 0 {
				nParam.Grad = node.Grad
				break
			}
		}
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
		weights = append(weights, NewValue(GenerateRandomFloat(-1, 1), "weight"))
	}
	bias := NewValue(GenerateRandomFloat(-1, 1), "bias")

	neuron := &Neuron{
		Weights: weights,
		Bias:    bias,
	}

	return neuron
}

func (n Neuron) Call(x []*Value) *Value {
	act := n.Bias

	for i, _ := range x {
		act = act.Add(n.Weights[i].Mul(x[i], "xi*wi"), "xi*wi+b")
	}

	out := act.Tanh("tanh(wi*xi+b)")
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

func (l Layer) Call(x []*Value) []*Value {
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

	for _, el := range x {
		l = append(l, NewValue(el, ""))
	}

	for _, layer := range mlp.Layers {
		l = layer.Call(l)
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
	// Training loop
	xs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}

	ys := []float64{1.0, -1.0, -1.0, 1.0}

	n := NewMLP(3, []int{4, 4, 1})

	for i := 0; i < 30; i++ {
		var ypred []*Value
		loss := NewValue(0.0, "Initial loss value")

		for _, x := range xs {
			ypred = append(ypred, n.Call(x))
		}

		for i, _ := range ypred {
			ygt := NewValue(ys[i], "ygt")
			yout := ypred[i]

			loss = loss.Add(
				(yout.Sub(ygt, "yout-ygt")).Pow(2, "Pow for mse"), "Add to loss")
		}

		for _, p := range n.Parameters() {
			p.Grad = 0.0
		}

		Backward(loss, n)

		for _, p := range n.Parameters() {
			lr := -0.1
			p.Data += p.Grad * lr
		}

		fmt.Printf("Step: %d, Loss: %f\n", i, loss.Data)

		if i == 29 {
			fmt.Println("Final predictions:")
			for _, pred := range ypred {
				pred.Print()
			}
		}
	}

	// // Test basic backward works
	// x1 := NewValue(2.0, "x1")
	// x2 := NewValue(0.0, "x2")

	// w1 := NewValue(-3.0, "w1")
	// w2 := NewValue(1.0, "w2")

	// b := NewValue(6.8813735870195432, "b")

	// x1w1 := x1.Mul(w1, "x1*w1")

	// x2w2 := x2.Mul(w2, "x2*w2")

	// x1w1x2w2 := x1w1.Add(x2w2, "x1*w1 + x2*w2")

	// n := x1w1x2w2.Add(b, "x1*w1 + x2*w2 + b")
	// n.Label = "n"

	// o := n.Tanh("tanh(x1*w1 + x2*w2 +b)")
	// o.Label = "o"

	// o.Backward()
}
