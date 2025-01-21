package test

import (
	"fmt"
	"math"
	"math/big"
	"runtime"
	"sync"
	"testing"

	"github.com/lifejade/imag-merge-resnet/cnn"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
	"github.com/tuneinsight/lattigo/v5/utils/sampling"
)

func Test_Boot(t *testing.T) {
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	ckksParams := cnn.CNN_Cifar18_Parameters
	initparams, _ := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)

	bootParams15 := ckksParams.BootstrappingParams
	//bootParams14.LogSlots, bootParams13.LogSlots, bootParams12.LogSlots = utils.Pointy(14), utils.Pointy(13), utils.Pointy(12)
	btpParams15, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams15)
	if err != nil {
		panic(err)
	}
	btpParams15.SlotsToCoeffsParameters.Scaling = new(big.Float).SetFloat64(0.5)
	//bootParams14 := ckksParams.BootstrappingParams
	//btpParams14, err := bootstrapping.NewParametersFromLiteral(params,bootParams14)

	fmt.Println("bootstrapping parameter init end")
	//keytime := time.Now()
	initkgen := rlwe.NewKeyGenerator(initparams)
	initsk := initkgen.GenSecretKeyNew()

	var params hefloat.Parameters
	var pk *rlwe.PublicKey
	var sk *rlwe.SecretKey
	var rlk *rlwe.RelinearizationKey
	var rtk []*rlwe.GaloisKey

	parThreadNum := 16
	threadNum := 1
	var wg_th sync.WaitGroup
	wg_th.Add(parThreadNum)
	btparr := make([]*bootstrapping.Evaluator, parThreadNum)
	for i := 0; i < threadNum; i++ {
		//generate bootstrapper
		var btpevk15_ *bootstrapping.EvaluationKeys
		btpevk15_, sk, _ = btpParams15.GenEvaluationKeys(initsk)
		for j := 0; j < parThreadNum; j++ {
			go func() {
				defer wg_th.Done()
				btparr[j], _ = bootstrapping.NewEvaluator(btpParams15, btpevk15_)
			}()
		}
		wg_th.Wait()
		runtime.GC()
		btp15 := btparr[0]

		fmt.Println("generated bootstrapper end")
		if i == 0 {
			params = *btp15.GetParameters()
		}
		kgen := rlwe.NewKeyGenerator(params)
		if i == 0 {
			pk = kgen.GenPublicKeyNew(sk)
			rlk = kgen.GenRelinearizationKeyNew(sk)
			// generate keys - Rotating key
			convRot := []int{1, 2, 3, 4, 5, 6}
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
		}

	}
	evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
	encryptor := rlwe.NewEncryptor(params, pk)
	decryptor := rlwe.NewDecryptor(params, sk)
	encoder := hefloat.NewEncoder(params)
	evaluator := hefloat.NewEvaluator(params, evk)
	_ = evaluator
	fmt.Println("generate Evaluator end")

	cipher := make([]*rlwe.Ciphertext, 16)
	n := 1 << params.LogMaxSlots()
	for i, _ := range cipher {
		value := make([]complex128, n)
		for j, _ := range value {
			value[j] = sampling.RandComplex128(-0.3, 0.3)
		}
		plaintext1 := hefloat.NewPlaintext(params, 0)
		encoder.Encode(value, plaintext1)
		cipher[i], _ = encryptor.EncryptNew(plaintext1)
	}

	for i, _ := range cipher {
		cnn.DecryptPrint(params, cipher[i], *decryptor, *encoder)
	}
	fmt.Println(params.MaxLevel())
	var wg sync.WaitGroup
	wg.Add(parThreadNum)
	for i := 0; i < parThreadNum; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < len(cipher); j++ {
				if j%parThreadNum == i {
					fmt.Println(j, "th is boot")
					cipher[j], _ = btparr[i].Bootstrap(cipher[j])
				}
			}
		}()
	}
	wg.Wait()

	for i, _ := range cipher {
		cnn.DecryptPrint(params, cipher[i], *decryptor, *encoder)
	}
}

func Test_MultyBoot(t *testing.T) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))
	threadNum := 10

	//ckks parameter init
	ckksParams := cnn.CNN_Cifar18_Parameters
	initparams, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
	if err != nil {
		panic(err)
	}
	fmt.Println("ckks parameter init end")

	bootParams15 := ckksParams.BootstrappingParams
	//bootParams14.LogSlots, bootParams13.LogSlots, bootParams12.LogSlots = utils.Pointy(14), utils.Pointy(13), utils.Pointy(12)
	btpParams15, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams15)
	if err != nil {
		panic(err)
	}
	btpParams15.SlotsToCoeffsParameters.Scaling = new(big.Float).SetFloat64(0.5)
	//bootParams14 := ckksParams.BootstrappingParams
	//btpParams14, err := bootstrapping.NewParametersFromLiteral(params,bootParams14)
	fmt.Println("bootstrapping parameter init end")

	// generate keys
	//fmt.Println("generate keys")
	//keytime := time.Now()
	initkgen := rlwe.NewKeyGenerator(initparams)
	initsk := initkgen.GenSecretKeyNew()

	var params hefloat.Parameters
	var pk *rlwe.PublicKey
	var sk *rlwe.SecretKey
	var btpevk *bootstrapping.EvaluationKeys
	var evk *rlwe.MemEvaluationKeySet

	btpevk, sk, _ = btpParams15.GenEvaluationKeys(initsk)
	btp, _ := bootstrapping.NewEvaluator(btpParams15, btpevk)
	params = *btp.GetParameters()
	kgen := rlwe.NewKeyGenerator(params)
	pk = kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)
	galEls := make([]uint64, 16)
	rts := make(map[int]struct{})
	for i := range 16 {
		r := int(math.Pow(2, float64(i)))
		galEls[i] = params.GaloisElement(r)
		rts[r] = struct{}{}
	}
	galEls = append(galEls, params.GaloisElementForComplexConjugation())
	rtk := make([]*rlwe.GaloisKey, len(galEls))

	var wg sync.WaitGroup
	wg.Add(len(galEls))
	for i := range galEls {
		go func() {
			defer wg.Done()
			kgen_ := rlwe.NewKeyGenerator(params)
			rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
		}()
	}
	wg.Wait()
	evk = rlwe.NewMemEvaluationKeySet(rlk, rtk...)

	encryptors := make([]*rlwe.Encryptor, threadNum)
	var decryptor *rlwe.Decryptor
	var encoder *hefloat.Encoder
	evaluators := make([]*hefloat.Evaluator, threadNum)
	btps := make([]*bootstrapping.Evaluator, threadNum)
	wg.Add(threadNum)
	for i := 0; i < threadNum; i++ {
		go func() {
			defer wg.Done()
			//generate bootstrapper
			if i == 0 {
				btps[i] = btp
				decryptor = rlwe.NewDecryptor(params, sk)
				encoder = hefloat.NewEncoder(params)
			} else {
				btps[i] = btp.ShallowCopy()
				// btps[i], _ = bootstrapping.NewEvaluator(btpParams15,btpevk)
			}
			//generate -er
			encryptors[i] = rlwe.NewEncryptor(params, pk)
			evaluators[i] = hefloat.NewEvaluator(params, evk)
		}()
	}
	wg.Wait()

	cipher := make([]*rlwe.Ciphertext, 5)
	n := 1 << params.LogMaxSlots()
	for i := range cipher {
		value := make([]float64, n)
		for j := range value {
			if (i+j)%3 == 0 {
				value[j] = sampling.RandFloat64(0.001, 0.003)
			} else {
				value[j] = -sampling.RandFloat64(0.001, 0.003)
			}

		}
		plaintext1 := hefloat.NewPlaintext(params, 0)
		encoder.Encode(value, plaintext1)
		cipher[i], _ = encryptors[0].EncryptNew(plaintext1)
	}

	result := make([]*rlwe.Ciphertext, len(cipher))
	wg.Add(threadNum)
	for threadidx := range threadNum {

		go func() {
			defer wg.Done()
			btp := btps[threadidx]
			fmt.Println(btp.BootstrappingParameters.N())
			for i := range cipher {
				if (i-threadidx)%threadNum != 0 {
					continue
				}
				var err error
				result[i], err = btp.Bootstrap(cipher[i])
				if err != nil {
					fmt.Println(i, "th: ", err)
				}
			}
		}()
	}
	wg.Wait()

	for i := range cipher {
		cnn.DecryptPrint(params, result[i], *decryptor, *encoder)
	}
}
