package engine

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

func NewValue(data float64) *Value {
	value := &Value{
		Data:      data,
		Grad:      0.0,
		Label:     "",
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

func (v1 *Value) Add(v2 *Value) *Value {
	out := NewValue(v1.Data + v2.Data)
	out.Op = Add
	out.Prev = []*Value{v1, v2}

	out._Backward = func(out *Value) {
		v1.Grad += 1.0 * out.Grad
		v2.Grad += 1.0 * out.Grad
	}

	return out
}

func (v1 *Value) Mul(v2 *Value) *Value {
	out := NewValue(v1.Data * v2.Data)
	out.Op = Mul
	out.Prev = []*Value{v1, v2}

	out._Backward = func(out *Value) {
		v1.Grad += v2.Data * out.Grad
		v2.Grad += v1.Data * out.Grad
	}

	return out
}

func (v1 *Value) Neg() *Value {
	v2 := NewValue(-1)
	out := NewValue(v1.Data * v2.Data)
	out.Op = Neg
	out.Prev = []*Value{v1, v2}

	return out
}

func (v1 *Value) Sub(v2 *Value) *Value {
	out := v1.Add(v2.Neg())
	out.Op = Sub
	out.Prev = []*Value{v1, v2}

	return out
}

func (v1 *Value) Pow(v2 float64) *Value {
	out := NewValue(float64(math.Pow(v1.Data, v2)))
	out.Op = Pow
	out.Prev = []*Value{v1}

	out._Backward = func(out *Value) {
		v1.Grad += (v2 * math.Pow(v1.Data, v2-1)) * out.Grad
	}

	return out
}

func (v1 *Value) Div(v2 *Value) *Value {
	out := NewValue(v1.Data * v2.Pow(-1).Data)
	out.Op = Div
	out.Prev = []*Value{v1, v2}

	return out
}

func (v *Value) Tanh() *Value {
	x := v.Data
	t := (math.Exp(2*x) - 1) / (math.Exp(2*x) + 1)
	out := NewValue(t)
	out.Op = Tanh
	out.Prev = []*Value{v}

	out._Backward = func(out *Value) {
		v.Grad += (1 - math.Pow(t, 2)) * out.Grad
	}

	return out
}

func (v *Value) Exp() *Value {
	x := v.Data
	out := NewValue(math.Exp(x))
	out.Op = Exp
	out.Prev = []*Value{v}

	out._Backward = func(out *Value) {
		v.Grad += out.Data * out.Grad
	}

	return out
}

func (v *Value) Backward() {
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
}
