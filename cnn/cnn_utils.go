package cnn

import (
	"math"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
)

func ctZero(context *Context, threadidx int) *rlwe.Ciphertext {
	n := context.params_.LogMaxSlots()
	zero := make([]complex128, n)
	plain := hefloat.NewPlaintext(*context.params_, context.params_.MaxLevel())
	context.encoders_[threadidx].Encode(zero, plain)
	ct, err := context.encryptors_[threadidx].EncryptNew(plain)
	if err != nil {
		panic(err)
	}
	return ct
}

func sumSlot(input *rlwe.Ciphertext, addSize, gap int, context *Context, threadidx int) *rlwe.Ciphertext {
	eval := context.evals_[threadidx]
	out := input.CopyNew()
	sum := ctZero(context, threadidx)
	logsize := int(math.Log2(float64(addSize)))

	for i := 0; i < logsize; i++ {
		if int(addSize/int(math.Pow(2, float64(i))))%2 == 1 {
			r := int(addSize/int(math.Pow(2, float64(i+1)))) * int(math.Pow(2, float64(i+1))) * gap
			temp := rotCtNew(out, r, context, threadidx)
			eval.Add(sum, temp, sum)
		}
		r := int(math.Pow(2, float64(i))) * gap
		temp := rotCtNew(out, r, context, threadidx)
		eval.Add(out, temp, out)
	}
	eval.Add(out, sum, out)
	return out
}

func rotCtNew(input *rlwe.Ciphertext, r int, context *Context, threadidx int) *rlwe.Ciphertext {
	rts := *context.rts_
	eval := context.evals_[threadidx]
	if r < 0 {
		mod := 1 << context.params_.LogMaxSlots()
		r += mod
	}

	if _, exist := rts[r]; exist {
		result, _ := eval.RotateNew(input, r)
		return result
	}
	//else

	result := input.CopyNew()

	for it_r, i := r, 1; it_r > 0; {
		if it_r%2 == 1 {
			eval.Rotate(result, i, result)
		}
		it_r = it_r >> 1
		i = i << 1
	}

	return result
}
