package main

import (
	"fmt"
	"math"
)

type Value struct {
	Data  float64
	Grad  float64
	Label string
	Op    Operation
	Prev  []*Value
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
		Data:  data,
		Grad:  0.0,
		Label: label,
		Op:    Nil,
		Prev:  nil,
	}

	return value
}

func (v Value) Print() {
	fmt.Printf("Value(data=%f)\n", v.Data)
}

func (v1 Value) Add(v2 *Value) *Value {
	out := NewValue(v1.Data+v2.Data, "")
	out.Op = Add
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) Mul(v2 *Value) *Value {
	out := NewValue(v1.Data*v2.Data, "")
	out.Op = Mul
	out.Prev = []*Value{&v1, v2}

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

func (v1 Value) Pow(v2 float64) *Value {
	out := NewValue(float64(math.Pow(v1.Data, v2)), "")
	out.Op = Pow
	out.Prev = []*Value{&v1}

	return out
}

func (v1 Value) Div(v2 *Value) *Value {
	out := NewValue(v1.Data*v2.Pow(-1).Data, "")
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

	return out
}

func main() {
	h := 0.0001

	a := NewValue(2.0, "a")
	b := NewValue(-3.0, "b")
	c := NewValue(10.0, "c")
	e := a.Mul(b)
	e.Label = "e"
	d := e.Add(c)
	d.Label = "d"
	f := NewValue(-2.0, "f")
	L := d.Mul(f)
	L.Label = "L"
	L1 := L.Data

	a = NewValue(2.0, "a")
	b = NewValue(-3.0, "b")
	c = NewValue(10.0, "c")
	e = a.Mul(b)
	e.Label = "e"
	d = e.Add(c)
	d.Label = "d"
	f = NewValue(-2.0, "f")
	L = d.Mul(f)
	L.Label = "L"
	L2 := L.Data + h

	fmt.Println((L2 - L1) / h)
}
