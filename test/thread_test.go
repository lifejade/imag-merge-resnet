package test

import (
	"fmt"
	"runtime"
	"sync"
	"testing"

	"github.com/lifejade/imag-merge-resnet/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_Multy(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	threadNum := 128

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
	var wg_ sync.WaitGroup
	wg_.Add(len(galEls))
	for i := range galEls {
		i := i
		go func() {
			defer wg_.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg_.Wait()
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	//generate -er
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	_, _, _, _ = encoder, encryptor, decryptor, evaluator

	fmt.Println("generate Evaluator end")

	encryptors := make([]*rlwe.Encryptor, threadNum)
	evaluators := make([]*hefloat.Evaluator, threadNum)
	for i := range threadNum {
		encryptors[i] = rlwe.NewEncryptor(params, pk)
		evaluators[i] = hefloat.NewEvaluator(params, evk)
	}

	cipher := make([]*rlwe.Ciphertext, threadNum)
	value := make([][]float64, threadNum)
	n := 1 << params.LogMaxSlots()
	for i := range value {
		value := make([]float64, n)
		for j := range n {
			if (i+j)%3 == 0 {
				value[j] = sampling.RandFloat64(0.001, 0.003)
			} else {
				value[j] = -sampling.RandFloat64(0.001, 0.003)
			}

		}
	}
	fmt.Println("value init end")
	var wg sync.WaitGroup
	wg.Add(threadNum)
	for i := range threadNum {
		go func() {
			defer wg.Done()
			plaintext1 := hefloat.NewPlaintext(params, params.MaxLevel())
			encoder.Encode(value, plaintext1)
			cipher[i], _ = encryptors[i].EncryptNew(plaintext1)
			evaluators[i].Rotate(cipher[i], 3, cipher[i])
		}()
	}
	wg.Wait()

}
