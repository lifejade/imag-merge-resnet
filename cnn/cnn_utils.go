package cnn

import (
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func ctZero(context *Context) *rlwe.Ciphertext {
	n := context.params_.LogMaxSlots()
	zero := make([]complex128, n)
	plain := hefloat.NewPlaintext(*context.params_, context.params_.MaxLevel())
	context.encoder_.Encode(zero, plain)
	ct, err := context.encryptor_.EncryptNew(plain)
	if err != nil {
		panic(err)
	}
	return ct
}

func sumSlot(input *rlwe.Ciphertext, addSize, gap int, context *Context) *rlwe.Ciphertext {
	eval := context.eval_
	out := input.CopyNew()
	sum := ctZero(context)
	logsize := int(math.Log2(float64(addSize)))

	for i := 0; i < logsize; i++ {
		if int(addSize/int(math.Pow(2, float64(i))))%2 == 1 {
			r := int(addSize/int(math.Pow(2, float64(i+1)))) * int(math.Pow(2, float64(i+1))) * gap
			temp, _ := eval.RotateNew(out, r)
			eval.Add(sum, temp, sum)
		}
		r := int(math.Pow(2, float64(i))) * gap
		temp, _ := eval.RotateNew(out, r)
		eval.Add(out, temp, out)
	}
	eval.Add(out, sum, out)
	return out
}
