package nn

import (
	"math"
	"math/rand"
	"time"

	"github.com/farrelmahaztra/micrograd-go/pkg/engine"
)

func GenerateRandomFloat(min float64, max float64) float64 {
	rand.Seed(time.Now().UnixNano())

	upper := math.Ceil(min)
	lower := math.Floor(max)
	n := rand.Float64()*(upper-lower) + lower

	return n
}

type Neuron struct {
	Weights []*engine.Value
	Bias    *engine.Value
}

func NewNeuron(nin int) *Neuron {
	var weights []*engine.Value
	for i := 0; i < nin; i++ {
		weights = append(weights, engine.NewValue(GenerateRandomFloat(-1, 1), "weight"))
	}
	bias := engine.NewValue(GenerateRandomFloat(-1, 1), "bias")

	neuron := &Neuron{
		Weights: weights,
		Bias:    bias,
	}

	return neuron
}

func (n Neuron) Call(x []*engine.Value) *engine.Value {
	act := n.Bias

	for i, _ := range x {
		act = act.Add(n.Weights[i].Mul(x[i], "xi*wi"), "xi*wi+b")
	}

	out := act.Tanh("tanh(wi*xi+b)")
	return out
}

func (n Neuron) Parameters() []*engine.Value {
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

func (l Layer) Call(x []*engine.Value) []*engine.Value {
	var outs []*engine.Value

	for _, n := range l.Neurons {
		outs = append(outs, n.Call(x))
	}

	return outs
}

func (l Layer) Parameters() []*engine.Value {
	var params []*engine.Value

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

func (mlp MLP) Call(x []float64) *engine.Value {
	var l []*engine.Value

	for _, el := range x {
		l = append(l, engine.NewValue(el, ""))
	}

	for _, layer := range mlp.Layers {
		l = layer.Call(l)
	}

	return l[0]
}

func (mlp MLP) Parameters() []*engine.Value {
	var params []*engine.Value

	for _, layer := range mlp.Layers {
		params = append(params, layer.Parameters()...)
	}

	return params
}
