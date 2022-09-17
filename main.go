package main

import (
	"fmt"
	"math"
)

type Value struct {
	Data  float64
	Grad  float64
	Label string
	Op    string
	Prev  []*Value
}

func NewValue(data float64, label string) *Value {
	value := &Value{
		Data:  data,
		Grad:  0.0,
		Label: label,
		Op:    "",
		Prev:  nil,
	}

	return value
}

func (v Value) Print() {
	fmt.Printf("Value(data=%f)\n", v.Data)
}

func (v1 Value) Add(v2 *Value) *Value {
	out := NewValue(v1.Data+v2.Data, "")
	out.Op = "+"
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) Mul(v2 *Value) *Value {
	out := NewValue(v1.Data*v2.Data, "")
	out.Op = "*"
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) Neg() *Value {
	v2 := NewValue(-1, "")
	out := NewValue(v1.Data*v2.Data, "")
	out.Op = "*-1"
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) Sub(v2 *Value) *Value {
	out := v1.Add(v2.Neg())
	out.Op = "-"

	return out
}

func (v1 Value) Pow(v2 float64) *Value {
	out := NewValue(float64(math.Pow(v1.Data, v2)), "")
	out.Op = "**"
	out.Prev = []*Value{&v1}

	return out
}

func (v1 Value) Div(v2 *Value) *Value {
	out := NewValue(v1.Data*v2.Pow(-1).Data, "")
	out.Op = "/"
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v Value) Tanh() *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := NewValue(t, "")
	out.Op = "tanh"
	out.Prev = []*Value{&v}

	return out
}

func (v Value) Exp() *Value {
	x := v.Data
	out := NewValue(math.Exp(x), "")
	out.Op = "e^"
	out.Prev = []*Value{&v}

	return out
}

func main() {
	x1 := NewValue(1, "x1")
	w1 := NewValue(1.5, "w1")

	x1w1 := x1.Mul(w1)
	x1w1.Label = "x1*w1"

	x1w1.Exp().Print()
}
