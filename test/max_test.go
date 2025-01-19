package test

import (
	"fmt"
	"os"
	"runtime"
	"sync"
	"testing"

	"github.com/lifejade/imag-merge-resnet/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_Maxfunc(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	layerNum := 18
	//check layernumber
	if !(layerNum == 18 || layerNum == 32) {
		fmt.Println("layer_num is not correct")
		os.Exit(1)
	}

	//ckks parameter init
	ckksParams := cnn.CNN_Cifar18_Parameters
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)
	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)

	fmt.Println("generate Evaluator end")
	context := cnn.NewContext(encoder, encryptor, decryptor, sk, pk, nil, rtk, rlk, evaluator, &params)

	cipher := make([]*rlwe.Ciphertext, 2)
	n := 1 << params.LogMaxSlots()
	for i, _ := range cipher {
		value := make([]float64, n)
		for j := range value {
			if (i+j)%3 == 0 {
				value[j] = sampling.RandFloat64(0.001, 0.003)
			} else {
				value[j] = -sampling.RandFloat64(0.001, 0.003)
			}

		}
		plaintext1 := hefloat.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(value, plaintext1)
		cipher[i], _ = encryptor.EncryptNew(plaintext1)
	}

	for i := range cipher {
		cnn.DecryptPrint(params, cipher[i], *decryptor, *encoder)
	}
	fmt.Println(params.MaxLevel())

	result := cnn.EvalApproxMinimaxMax(cipher[0], cipher[1], 14, context)
	cnn.DecryptPrint(params, result, *decryptor, *encoder)
}

func Test_Maxfunc3(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	layerNum := 18
	//check layernumber
	if !(layerNum == 18 || layerNum == 32) {
		fmt.Println("layer_num is not correct")
		os.Exit(1)
	}

	//ckks parameter init
	ckksParams := cnn.CNN_Cifar18_Parameters
	params, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()

	var pk *rlwe.PublicKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	fmt.Println("generated bootstrapper end")
	pk = kgen.GenPublicKeyNew(sk)
	rlk = kgen.GenRelinearizationKeyNew(sk)
	// generate keys - Rotating key
	convRot := []int{0, 1, 2, 3, 4, 5}
	galEls := make([]uint64, len(convRot))
	for i, x := range convRot {
		galEls[i] = params.GaloisElement(x)
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())

	rtk = make([]*rlwe.GaloisKey, len(galEls))
	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		i := i
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)

	fmt.Println("generate Evaluator end")
	context := cnn.NewContext(encoder, encryptor, decryptor, sk, pk, nil, rtk, rlk, evaluator, &params)

	cipher := make([]*rlwe.Ciphertext, 3)
	n := 1 << params.LogMaxSlots()
	for i, _ := range cipher {
		value := make([]float64, n)
		for j := range value {
			if (i+j)%3 == 0 {
				value[j] = sampling.RandFloat64(0.01, 0.03)
			} else {
				value[j] = -sampling.RandFloat64(0.01, 0.03)
			}
		}
		plaintext1 := hefloat.NewPlaintext(params, params.MaxLevel())
		encoder.Encode(value, plaintext1)
		cipher[i], _ = encryptor.EncryptNew(plaintext1)
	}

	for i := range cipher {
		cnn.DecryptPrint(params, cipher[i], *decryptor, *encoder)
	}
	fmt.Println(params.MaxLevel())

	result := cnn.EvalApproxMinimaxMax3(cipher[0], cipher[1], cipher[2], 14, context)
	cnn.DecryptPrint(params, result, *decryptor, *encoder)
}
