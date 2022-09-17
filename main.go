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

func (v1 Value) Pow(v2 *Value) *Value {
	out := NewValue(float64(math.Pow(v1.Data, v2.Data)), "")
	out.Op = Pow
	out.Prev = []*Value{&v1, v2}

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

	return out
}

func (v Value) _Backward() {
	op := v.Op
	prev := v.Prev

	switch {
	case op == Add || op == Sub:
		prev[0].Grad += 1.0 * v.Grad
		prev[1].Grad += 1.0 * v.Grad

	case op == Mul || op == Div:
		prev[0].Grad += prev[1].Data * v.Grad
		prev[1].Grad += prev[0].Data * v.Grad

	case op == Pow:
		prev[0].Grad += (prev[1].Data * math.Pow(prev[0].Data, prev[1].Data-1)) * v.Grad
		prev[1].Grad += (v.Data * math.Log(prev[0].Data)) * v.Grad

	case op == Exp:
		prev[0].Grad += v.Data * v.Grad
	}
}

func main() {
	a := NewValue(3, "a")
	b := NewValue(2, "b")
	c := a.Pow(b)
	c.Label = "c"
	c.Grad = 1.0
	c._Backward()
	fmt.Println(c.Prev[0].Grad, c.Prev[1].Grad)

	//h := 0.0001

	// a := NewValue(2.0, "a")
	// b := NewValue(-3.0, "b")
	// c := NewValue(10.0, "c")
	// e := a.Mul(b)
	// e.Label = "e"
	// d := e.Add(c)
	// d.Label = "d"
	// f := NewValue(-2.0, "f")
	// L := d.Mul(f)
	// L.Label = "L"
	// L1 := L.Data

	// a = NewValue(2.0, "a")
	// b = NewValue(-3.0, "b")
	// c = NewValue(10.0, "c")
	// e = a.Mul(b)
	// e.Label = "e"
	// d = e.Add(c)
	// d.Label = "d"
	// f = NewValue(-2.0, "f")
	// L = d.Mul(f)
	// L.Label = "L"
	// L2 := L.Data

	//fmt.Println((L2 - L1) / h)

}
