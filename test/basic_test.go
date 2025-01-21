package test

import (
	"fmt"
	"testing"

	"github.com/lifejade/imag-merge-resnet/cnn"
)

type A struct {
	a []float64
}

func Test_Load(t *testing.T) {
	dir := "resnet18_new"
	temp := make([]A, 10)
	temp[0].a = []float64{}
	temp[0].a = make([]float64, 7*7*3*64)
	cnn.ReadConvWgtIdx("../parameters/resnet_pretrained/"+dir+"/conv1_weight.txt", &temp[0].a, 7*7*3*64)
	for _, v := range temp[0].a {
		t.Logf("%f\n", v)
	}
	t.Logf("%d\n", len(temp[0].a))
}

func Test_Mod(t *testing.T) {
	fmt.Println(-61 % 3)
}
