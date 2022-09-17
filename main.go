package main

import "fmt"

type Value struct {
	Data  float32
	Grad  float32
	Label string
	Op    string
	Prev  []*Value
}

func NewValue(data float32, label string) *Value {
	value := &Value{
		Data:  data,
		Grad:  0.0,
		Label: label,
		Op:    "",
		Prev:  nil,
	}

	return value
}

func (v1 Value) AddValue(v2 *Value) *Value {
	out := NewValue(v1.Data+v2.Data, "")
	out.Op = "+"
	out.Prev = []*Value{&v1, v2}

	return out
}

func main() {
	a := NewValue(1.0, "a")
	b := NewValue(2.0, "b")
	c := a.AddValue(b)
	c.Label = "a+b"
	fmt.Println(c)
}
