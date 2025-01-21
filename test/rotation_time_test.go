package test

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/lifejade/imag-merge-resnet/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_RotTime(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

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
	galEls := make([]uint64, 16)
	for i := range 16 {
		galEls[i] = params.GaloisElement(int(math.Pow(2, float64(i))))
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
	n := 1 << params.LogMaxSlots()
	value := make([]float64, n)
	for i := range n {
		if (i)%3 == 0 {
			value[i] = sampling.RandFloat64(0.001, 0.003)
		} else {
			value[i] = -sampling.RandFloat64(0.001, 0.003)
		}
	}
	fmt.Println("value init end")

	plaintext1 := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext1)
	cipher, _ := encryptor.EncryptNew(plaintext1)

	cnn.DecryptPrint(params, cipher, *decryptor, *encoder)
	startTime := time.Now()
	rot, _ := evaluator.RotateNew(cipher, 16)
	elapse := time.Since(startTime)
	fmt.Println(elapse)
	cnn.DecryptPrint(params, rot, *decryptor, *encoder)
	fmt.Println("//////////////////////////////////////")
	rot2 := cipher.CopyNew()
	startTime = time.Now()
	for i := range 16 {
		_ = i
		evaluator.Rotate(rot2, 1, rot2)
	}
	elapse = time.Since(startTime)
	fmt.Println(elapse)
	cnn.DecryptPrint(params, rot2, *decryptor, *encoder)
}

func Test_RotHoistTime(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

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
	galEls := make([]uint64, 16)
	for i := range 16 {
		galEls[i] = params.GaloisElement(int(math.Pow(2, float64(i))))
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
	n := 1 << params.LogMaxSlots()
	value := make([]float64, n)
	for i := range n {
		if (i)%3 == 0 {
			value[i] = sampling.RandFloat64(0.001, 0.003)
		} else {
			value[i] = -sampling.RandFloat64(0.001, 0.003)
		}
	}
	plaintext1 := hefloat.NewPlaintext(params, params.MaxLevel())
	encoder.Encode(value, plaintext1)
	cipher, _ := encryptor.EncryptNew(plaintext1)
	cnn.DecryptPrint(params, cipher, *decryptor, *encoder)

	rtc := make([]*rlwe.Ciphertext, 4)
	arr := []int{1, 2, 4, 8}
	startTime := time.Now()
	for i := range rtc {
		rtc[i], _ = evaluator.RotateNew(cipher, arr[i])
	}
	elapse := time.Since(startTime)
	fmt.Println(elapse)
	for i := range rtc {
		cnn.DecryptPrint(params, rtc[i], *decryptor, *encoder)
	}

	fmt.Println("//////////////////////////////////////")

	startTime = time.Now()
	rtc2, _ := evaluator.RotateHoistedNew(cipher, arr)
	elapse = time.Since(startTime)
	fmt.Println(elapse)
	for i := range rtc2 {
		fmt.Println(i)
		cnn.DecryptPrint(params, rtc2[i], *decryptor, *encoder)
	}

	for i := range rtk {
		fmt.Println(rtk[i].BaseTwoDecomposition)
		fmt.Println(rtk[i].NthRoot)
	}

}
