package cnn

import (
	"fmt"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/ring"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test() {

	ckksparam := hefloat.ParametersLiteral{
		LogN:            16,
		LogQ:            []int{51, 46, 46, 46, 46, 46, 46, 51},
		LogP:            []int{51, 51, 51},
		Xs:              ring.Ternary{H: 192},
		LogDefaultScale: 46,
	}

	//parameter init
	params, err := hefloat.NewParametersFromLiteral(ckksparam)
	if err != nil {
		panic(err)
	}

	// generate classes
	kgen := rlwe.NewKeyGenerator(params)

	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)

	n := 1 << params.LogMaxSlots()
	rlk := kgen.GenRelinearizationKeyNew(sk)
	conRot := []int{0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3}
	galEls := make([]uint64, len(conRot))
	for i, x := range conRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())
	rtk := kgen.GenGaloisKeysNew(galEls, sk)

	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	evaluator := hefloat.NewEvaluator(params, evk)

	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)

	_, _, _, _ = encryptor, decryptor, encoder, evaluator

	value := make([]complex128, n)
	for i := range value {
		value[i] = sampling.RandComplex128(-1, 1)
	}
	plaintext := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext)
	cipher, _ := encryptor.EncryptNew(plaintext)

	cipher2, _ := evaluator.ConjugateNew(cipher)
	result := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(cipher2), result)

	fmt.Printf("%6.10f %6.10f %6.10f %6.10f ... %6.10f %6.10f\n", value[0], value[1], value[2], value[3], value[n-2], value[n-1])
	fmt.Printf("%6.10f %6.10f %6.10f %6.10f ... %6.10f %6.10f\n", result[0], result[1], result[2], result[3], result[n-2], result[n-1])
}
