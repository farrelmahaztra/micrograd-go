package main

import (
	"fmt"
	"math"
)

type Value struct {
	Data      float64
	Grad      float64
	Label     string
	Op        Operation
	Prev      []*Value
	_Backward func()
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
		_Backward: func() {},
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

	out._Backward = func() {
		(&v1).Grad += 1.0 * out.Grad
		v2.Grad += 1.0 * out.Grad
	}

	return out
}

func (v1 Value) Mul(v2 *Value) *Value {
	out := NewValue(v1.Data*v2.Data, "")
	out.Op = Mul
	out.Prev = []*Value{&v1, v2}

	out._Backward = func() {
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

	out._Backward = func() {
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

	return out
}

func (v Value) Exp() *Value {
	x := v.Data
	out := NewValue(math.Exp(x), "")
	out.Op = Exp
	out.Prev = []*Value{&v}

	out._Backward = func() {
		(&v).Grad += out.Data * out.Grad
	}

	return out
}

func (v Value) Backward() {
	var topo []*Value
	var visited []*Value
	var BuildTopo func(v *Value)

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
		topo[i]._Backward()
		topo[i].Print()
	}
}

func main() {
	a := NewValue(3, "a")
	b := NewValue(5, "b")
	c := a.Add(b)
	c.Label = "c"
	d := NewValue(2, "d")
	e := c.Mul(d)
	e.Label = "e"
	e.Grad = 1.0
	e.Backward()
}
