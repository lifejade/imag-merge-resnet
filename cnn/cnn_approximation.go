package cnn

import (
	"github.com/lifejade/imag-merge-resnet/cnn/comp"
	"github.com/lifejade/imag-merge-resnet/cnn/scaleinv"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
)

func EvalApproxMinimaxReLU(cipherin *rlwe.Ciphertext, alpha int, context *Context, threadidx int) (destination *rlwe.Ciphertext) {

	if alpha == 13 {

		compNo := 3
		deg := []int{15, 15, 27}
		scaledVal := 1.6
		var tree []comp.Tree

		for i := 0; i < compNo; i++ {

			tr := comp.OddBaby(deg[i])
			tree = append(tree, *tr)
			// tr.Print()
		}

		scaleContext := scaleinv.ScaleContext{
			Encoder_:   context.encoders_[threadidx],
			Encryptor_: context.encryptors_[threadidx],
			Decryptor_: context.decryptor_,
			Eval_:      context.evals_[threadidx],
			Params_:    *context.params_,
		}

		destination = comp.MinimaxReLU(compNo, alpha, deg, tree, scaledVal, scaleContext, cipherin)

	} else if alpha == 14 {

		compNo := 3
		deg := []int{15, 27, 29}
		scaledVal := 1.6
		var tree []comp.Tree

		for i := 0; i < compNo; i++ {

			tr := comp.OddBaby(deg[i])
			tree = append(tree, *tr)
			// tr.Print()
		}

		scaleContext := scaleinv.ScaleContext{
			Encoder_:   context.encoders_[threadidx],
			Encryptor_: context.encryptors_[threadidx],
			Decryptor_: context.decryptor_,
			Eval_:      context.evals_[threadidx],
			Params_:    *context.params_,
		}

		destination = comp.MinimaxReLU(compNo, alpha, deg, tree, scaledVal, scaleContext, cipherin)

	}

	return destination
}

func EvalApproxMinimaxMax(cipherin1, cipherin2 *rlwe.Ciphertext, alpha int, context *Context) (destination *rlwe.Ciphertext) {

	if alpha == 13 {

		compNo := 3
		deg := []int{15, 15, 27}
		scaledVal := 1.6
		var tree []comp.Tree

		for i := 0; i < compNo; i++ {

			tr := comp.OddBaby(deg[i])
			tree = append(tree, *tr)
			// tr.Print()
		}

		scaleContext := scaleinv.ScaleContext{
			Encoder_:   context.encoders_[0],
			Encryptor_: context.encryptors_[0],
			Decryptor_: context.decryptor_,
			Eval_:      context.evals_[0],
			Params_:    *context.params_,
		}

		destination = comp.MinimaxMax(compNo, alpha, deg, tree, scaledVal, scaleContext, cipherin1, cipherin2)

	} else if alpha == 14 {

		compNo := 3
		deg := []int{15, 27, 29}
		scaledVal := 1.6
		var tree []comp.Tree

		for i := 0; i < compNo; i++ {

			tr := comp.OddBaby(deg[i])
			tree = append(tree, *tr)
			// tr.Print()
		}

		scaleContext := scaleinv.ScaleContext{
			Encoder_:   context.encoders_[0],
			Encryptor_: context.encryptors_[0],
			Decryptor_: context.decryptor_,
			Eval_:      context.evals_[0],
			Params_:    *context.params_,
		}

		destination = comp.MinimaxMax(compNo, alpha, deg, tree, scaledVal, scaleContext, cipherin1, cipherin2)

	}

	return destination
}

func EvalApproxMinimaxMax3(cipherin1, cipherin2, cipherin3 *rlwe.Ciphertext, alpha int, context *Context, threadidx int) (destination *rlwe.Ciphertext) {

	if alpha == 13 {

		compNo := 3
		deg := []int{15, 15, 27}
		scaledVal := 1.6
		var tree []comp.Tree

		for i := 0; i < compNo; i++ {

			tr := comp.OddBaby(deg[i])
			tree = append(tree, *tr)
			// tr.Print()
		}

		scaleContext := scaleinv.ScaleContext{
			Encoder_:   context.encoders_[threadidx],
			Encryptor_: context.encryptors_[threadidx],
			Decryptor_: context.decryptor_,
			Eval_:      context.evals_[threadidx],
			Params_:    *context.params_,
		}

		destination = comp.MinimaxMax_3I(compNo, alpha, deg, tree, scaledVal, scaleContext, cipherin1, cipherin2, cipherin3)

	} else if alpha == 14 {
		//level consume = 16
		compNo := 3
		deg := []int{15, 27, 29}
		scaledVal := 1.6
		var tree []comp.Tree

		for i := 0; i < compNo; i++ {

			tr := comp.OddBaby(deg[i])
			tree = append(tree, *tr)
			// tr.Print()
		}

		scaleContext := scaleinv.ScaleContext{
			Encoder_:   context.encoders_[threadidx],
			Encryptor_: context.encryptors_[threadidx],
			Decryptor_: context.decryptor_,
			Eval_:      context.evals_[threadidx],
			Params_:    *context.params_,
		}

		destination = comp.MinimaxMax_3I(compNo, alpha, deg, tree, scaledVal, scaleContext, cipherin1, cipherin2, cipherin3)

	}

	return destination
}
