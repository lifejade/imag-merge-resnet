package cnn

import (
	"bufio"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"

	"math"
)

type TensorCipher struct {
	k_       int
	h_       int
	w_       int
	c_       int
	m_       int
	logn_    int
	ciphers_ []*rlwe.Ciphertext
}

type WeightConvBN struct {
	convwgt []float64
	bnmean  []float64
	bnbias  []float64
	bnvar   []float64
	bnwgt   []float64
}

func compactGappedConvolutionPrint(cnn TensorCipher, co, st, fh, fw int, convWgt, bnVar, bnWgt []float64, epsilon float64, context *Context, stage int, log_writer *bufio.Writer) TensorCipher {
	timeStart := time.Now()
	cnnOut := compactGappedConvolution(cnn, co, st, fh, fw, convWgt, bnVar, bnWgt, epsilon, context, stage)
	elapse := time.Since(timeStart)
	fmt.Printf("===================================================================================\n")
	log_writer.WriteString("===================================================================================\n")

	fmt.Printf("time: %s \n", elapse)
	fmt.Print("convolution ", stage, " result\n")
	log_writer.WriteString("time: " + elapse.String() + "\n")
	log_writer.WriteString("convolution " + strconv.Itoa(stage) + " result\n")
	decryptPrintTxt(cnnOut.ciphers_, log_writer, context, 17)
	printParmsTxt(cnnOut, log_writer)

	fmt.Printf("===================================================================================\n")
	log_writer.WriteString("===================================================================================\n")
	log_writer.Flush()
	return cnnOut
}

func compactGappedConvolution(cnn TensorCipher, co, st, fh, fw int, convWgt, bnVar, bnWgt []float64, epsilon float64, context *Context, stage int) TensorCipher {
	hi, wi, ki, ci, mi := cnn.h_, cnn.w_, cnn.k_, cnn.c_, cnn.m_
	var ho, wo, ko, mo int
	if st == 1 {
		ho, wo, ko = hi, wi, ki
	} else if st == 2 {
		ho, wo, ko = hi/2, wi/2, ki*2
	} else {
		fmt.Println("error")
		os.Exit(1)
	}
	logn := cnn.logn_
	n := 1 << cnn.logn_
	eval := context.eval_

	_, _ = n, ci
	checkTime := time.Now()
	mo = (co + ko*ko - 1) / (ko * ko)
	ctxt := make([][][]*rlwe.Ciphertext, mi)
	for m := 0; m < mi; m++ {
		ctxt[m] = make([][]*rlwe.Ciphertext, fh)
		for h := 0; h < fh; h++ {
			ctxt[m][h] = make([]*rlwe.Ciphertext, fw)
			for w := 0; w < fw; w++ {
				r := ki*ki*wi*(h-(fh-1)/2) + ki*(w-(fw-1)/2)
				ctxt[m][h][w], _ = eval.RotateNew(cnn.ciphers_[m], r)

			}
		}
	}
	elapse := time.Since(checkTime)
	fmt.Printf("elapse - Rotate Input Data: %s \n", elapse)

	checkTime = time.Now()
	compactWeightVec := make([][][][][]float64, fh)
	for i1 := 0; i1 < fh; i1++ {
		compactWeightVec[i1] = make([][][][]float64, fw)
		for i2 := 0; i2 < fw; i2++ {
			compactWeightVec[i1][i2] = make([][][]float64, mi)
			for m := 0; m < mi; m++ {
				compactWeightVec[i1][i2][m] = make([][]float64, co)
				for i := 0; i < co; i++ {
					compactWeightVec[i1][i2][m][i] = make([]float64, n)
					for j := 0; j < n; j++ {
						i3 := (j / (ki * wi)) % (ki * hi)
						i4 := j % (ki * wi)
						check1 := (ki*(i3%ki)+(i4%ki) >= ci)

						temp := (i3 / ki) - (fh-1)/2 + i1
						check2 := (temp < 0 || temp > hi-1)

						temp = (i4 / ki) - (fw-1)/2 + i2
						check3 := (temp < 0 || temp > wi-1)
						if check1 || check2 || check3 {
							compactWeightVec[i1][i2][m][i][j] = 0
						} else {
							idx := fw*i1 + i2 + (fw*fh)*(ki*ki*m+ki*(i3%ki)+(i4%ki)) + (fw*fh*ci)*i
							compactWeightVec[i1][i2][m][i][j] = convWgt[idx]
						}
					}
				}
			}
		}
	}
	elapse = time.Since(checkTime)
	fmt.Printf("elapse - init CNN Weight: %s \n", elapse)

	checkTime = time.Now()
	selectOneVec := make([][]float64, ko*ko)
	for i := 0; i < ko*ko; i++ {
		bnValue := bnWgt[i] / math.Sqrt(bnVar[i]+epsilon)
		//temp
		bnValue = 1
		selectOneVec[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			i4 := j % (ko * wo)
			i3 := (j / (ko * wo)) % (ko * ho)
			if ko*(i3%ko)+(i4%ko) == i {
				selectOneVec[i][j] = bnValue
			} else {
				selectOneVec[i][j] = 0
			}
		}
	}
	elapse = time.Since(checkTime)
	fmt.Printf("elapse - init Select Vector: %s \n", elapse)

	ctzero := ctZero(context)
	initScale := cnn.ciphers_[0].Scale

	result_cipher := make([]*rlwe.Ciphertext, mo)
	for i := 0; i < mo; i++ {
		result_cipher[i] = ctzero.CopyNew()
	}

	for i3 := 0; i3 < co; i3++ {
		checkTime = time.Now()
		cta := ctzero.CopyNew()
		for m := 0; m < mi; m++ {
			ctb := ctzero.CopyNew()
			for i1 := 0; i1 < fh; i1++ {
				for i2 := 0; i2 < fw; i2++ {
					temp, _ := eval.MulNew(ctxt[m][i1][i2], compactWeightVec[i1][i2][m][i3])
					_ = eval.Add(ctb, temp, ctb)
				}
			}

			eval.RescaleTo(ctb, initScale, ctb)
			ctc := sumSlot(ctb, ki, 1, context)
			ctc = sumSlot(ctc, ki, ki*wi, context)
			eval.Add(cta, ctc, cta)

		}
		r := -int((i3%(ko*ko))/ko)*ko*wo - (i3 % ko)
		temp, _ := eval.RotateNew(cta, r)
		eval.Mul(temp, selectOneVec[i3%(ko*ko)], temp)
		eval.Add(result_cipher[int(i3/(ko*ko))], temp, result_cipher[int(i3/(ko*ko))])
		runtime.GC()
		elapse := time.Since(checkTime)
		fmt.Printf("elapse - %dth out channel Conv: %s \n", i3, elapse)
	}
	for i := 0; i < mo; i++ {
		eval.RescaleTo(result_cipher[i], initScale, result_cipher[i])
	}
	result := NewTensorCipher(ko, ho, wo, co, mo, logn, result_cipher)
	return result
}

func compactGappedBatchNormPrint(cnn TensorCipher, bnbias, bnmean, bnvar, bnwgt []float64, epsilon, B float64, context *Context, stage int, log_writer *bufio.Writer) TensorCipher {
	return cnn
}
func approxReLUPrint(cnn TensorCipher, alpha int, log_writer *bufio.Writer, context *Context, stage int) TensorCipher {
	return cnn
}

func maxPoolingPrint(cnn TensorCipher, st, fh, fw int, context *Context, stage int, log_writer *bufio.Writer) TensorCipher {
	return cnn
}

func bootstrapPrint(cnn TensorCipher, context *Context, stage int, log_writer *bufio.Writer) TensorCipher {
	return cnn
}

func cipherAddPrint(cnn TensorCipher, adder TensorCipher, context *Context, stage int, log_writer *bufio.Writer) TensorCipher {
	return cnn
}

func averagepoolingPrint(cnn TensorCipher, B float64, context *Context, log_writer *bufio.Writer) TensorCipher {
	return cnn
}

func fullyConnectedPrint(cnn TensorCipher, linWgt, linBias []float64, outNum, ci int, context *Context, log_writer *bufio.Writer) TensorCipher {
	return cnn
}

func NewTensorCipherFormData(k, h, w, c, m, logn, logq int, data [][]float64, context *Context) TensorCipher {
	ciphers := make([]*rlwe.Ciphertext, c)
	for i := 0; i < c; i++ {
		plaintext := hefloat.NewPlaintext(*context.params_, context.params_.MaxLevel())
		plaintext.Scale = rlwe.NewScale(math.Pow(2.0, float64(logq)))
		context.encoder_.Encode(data[i], plaintext)
		cipher, err := context.encryptor_.EncryptNew(plaintext)
		if err != nil {
			panic(err)
		}
		ciphers[i] = cipher
	}
	return NewTensorCipher(k, h, w, c, m, logn, ciphers)
}

func NewTensorCipher(k, h, w, c, m, logn int, ciphers []*rlwe.Ciphertext) TensorCipher {
	result := TensorCipher{
		k_:       k,
		h_:       h,
		w_:       w,
		c_:       c,
		m_:       m,
		logn_:    logn,
		ciphers_: ciphers,
	}
	return result
}

func CopyTensorCipher(input TensorCipher) TensorCipher {
	ciphers := make([]*rlwe.Ciphertext, input.m_)
	for i := 0; i < input.m_; i++ {
		ciphers[i] = input.ciphers_[i].CopyNew()
	}
	result := TensorCipher{
		k_:       input.k_,
		h_:       input.h_,
		w_:       input.w_,
		c_:       input.c_,
		m_:       input.m_,
		logn_:    input.logn_,
		ciphers_: ciphers,
	}
	return result
}

type Context struct {
	encoder_   *hefloat.Encoder
	encryptor_ *rlwe.Encryptor
	decryptor_ *rlwe.Decryptor
	sk_        *rlwe.SecretKey
	pk_        *rlwe.PublicKey
	btp15_     *bootstrapping.Evaluator
	rotkeys_   []*rlwe.GaloisKey
	rlk_       *rlwe.RelinearizationKey
	eval_      *hefloat.Evaluator
	params_    *hefloat.Parameters
}

func NewContext(encoder *hefloat.Encoder, encryptor *rlwe.Encryptor, decryptor *rlwe.Decryptor, sk *rlwe.SecretKey,
	pk *rlwe.PublicKey, btp15 *bootstrapping.Evaluator, rotkeys []*rlwe.GaloisKey, rlk *rlwe.RelinearizationKey,
	eval *hefloat.Evaluator, params *hefloat.Parameters) *Context {
	result := Context{
		encoder_:   encoder,
		encryptor_: encryptor,
		decryptor_: decryptor,
		sk_:        sk,
		pk_:        pk,
		btp15_:     btp15,
		rotkeys_:   rotkeys,
		rlk_:       rlk,
		eval_:      eval,
		params_:    params,
	}
	return &result
}
func decryptPrint(ciphertexts []*rlwe.Ciphertext, context *Context, num int) {
	params := *context.params_
	decryptor := *context.decryptor_
	encoder := *context.encoder_
	// n := params.Slots()
	valuesTest := make([]complex128, params.LogMaxSlots())
	fmt.Print("/////////////////////////////////////////////////////////////////////\n")
	m := len(ciphertexts)
	for index := 0; index < m; index++ {
		encoder.Decode(decryptor.DecryptNew(ciphertexts[index]), valuesTest)
		fmt.Printf("%d ciphertext\n", index)
		fmt.Printf("Level: %d (logQ = %d)\n", ciphertexts[index].Level(), params.LogQLvl(ciphertexts[index].Level()))
		fmt.Printf("Scale: 2^%f\n", ciphertexts[index].LogScale())
		fmt.Printf("ValuesTest: ")
		for i := 0; i < num; i++ {
			fmt.Printf("%6.10f ", valuesTest[i])
			if (i+1)%32 == 0 {
				fmt.Println()
			}
		}
	}
	fmt.Printf("\n/////////////////////////////////////////////////////////////////////\n")
}
func decryptPrintTxt(ciphertexts []*rlwe.Ciphertext, output *bufio.Writer, context *Context, num int) {
	params := *context.params_
	decryptor := *context.decryptor_
	encoder := *context.encoder_
	// n := params.Slots()
	valuesTest := make([]complex128, params.MaxSlots())
	fmt.Print("/////////////////////////////////////////////////////////////////////\n")
	output.WriteString("/////////////////////////////////////////////////////////////////////\n")
	m := len(ciphertexts)
	for index := 0; index < m; index++ {
		encoder.Decode(decryptor.DecryptNew(ciphertexts[index]), valuesTest)
		fmt.Printf("%d ciphertext\n", index)
		fmt.Printf("Level: %d (logQ = %d)\n", ciphertexts[index].Level(), params.LogQLvl(ciphertexts[index].Level()))
		fmt.Printf("Scale: 2^%f\n", ciphertexts[index].LogScale())
		fmt.Printf("ValuesTest: ")
		for i := 0; i < num; i++ {
			fmt.Printf("%6.10f ", valuesTest[i])
			if (i+1)%32 == 0 {
				fmt.Println()
			}
		}

		output.WriteString(strconv.Itoa(index) + " ciphertext\n")
		output.WriteString("Level:" + strconv.Itoa(ciphertexts[index].Level()) + " (logQ = " + strconv.Itoa(params.LogQLvl(ciphertexts[index].Level())) + ")\n")
		output.WriteString("Scale: 2^" + fmt.Sprintf("%f", ciphertexts[index].LogScale()) + "\n")
		output.WriteString("ValuesTest: ")
		for i := 0; i < num; i++ {
			output.WriteString(fmt.Sprintf("%6.10f ", valuesTest[i]))
			if (i+1)%32 == 0 {
				output.WriteString("\n")
			}
		}
		if index != m-1 {
			fmt.Print("\n\n")
			output.WriteString("\n\n")
		}
	}

	fmt.Printf("\n/////////////////////////////////////////////////////////////////////\n")
	output.WriteString("\n/////////////////////////////////////////////////////////////////////\n")
	output.Flush()
}
func printParmsTxt(cnn TensorCipher, log_writer *bufio.Writer) {
	str := fmt.Sprintln("parameters: k:", cnn.k_, ", h:", cnn.h_, ", w:", cnn.w_, ", c:", cnn.c_, ", m:", cnn.m_, ", logn:", cnn.logn_)
	fmt.Print(str)
	log_writer.WriteString(str)
	log_writer.Flush()
}
