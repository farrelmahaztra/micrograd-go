package main

import (
	"fmt"
)

type Value struct {
	Data  float32
	Grad  float32
	Label string
	Op    string
	Prev  []*Value
}

func NewValue(data float32) *Value {
	value := &Value{
		Data:  data,
		Grad:  0.0,
		Label: "",
		Op:    "",
		Prev:  nil,
	}

	return value
}

func main() {
	x := NewValue(1.0)
	fmt.Println(x)
}
