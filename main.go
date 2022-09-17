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

func (v Value) PrintValue() {
	fmt.Printf("Value(data=%f)\n", v.Data)
}

func (v1 Value) AddValue(v2 *Value) *Value {
	out := NewValue(v1.Data+v2.Data, "")
	out.Op = "+"
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) MulValue(v2 *Value) *Value {
	out := NewValue(v1.Data*v2.Data, "")
	out.Op = "*"
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) NegValue() *Value {
	v2 := NewValue(-1, "")
	out := NewValue(v1.Data*v2.Data, "")
	out.Op = "*-1"
	out.Prev = []*Value{&v1, v2}

	return out
}

func (v1 Value) SubValue(v2 *Value) *Value {
	out := v1.AddValue(v2.NegValue())
	out.Op = "-"

	return out
}

func (v1 Value) PowValue(v2 *Value) *Value {
	out := NewValue(float64(math.Pow(v1.Data, v2.Data)), "")
	out.Op = "**"
	out.Prev = []*Value{&v1, v2}

	return out
}

func main() {
	x1 := NewValue(2.0, "x1")
	w1 := NewValue(-3.0, "w1")

	x1w1 := x1.MulValue(w1)
	x1w1.Label = "x1*w1"

	x2w2 := NewValue(2.0, "x2w2")
	x1w1.PowValue(x2w2).PrintValue()
}
