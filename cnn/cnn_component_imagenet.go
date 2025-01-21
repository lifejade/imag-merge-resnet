package cnn

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"sync"
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
	cnnOut := compactGappedConvolution(cnn, co, st, fh, fw, convWgt, bnVar, bnWgt, epsilon, context)
	elapse := time.Since(timeStart)

	str := fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("time: %s \n", elapse)
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("convolution %d-stage result\n", stage)
	fmt.Print(str)
	log_writer.WriteString(str)

	decryptPrintTxt(cnnOut.ciphers_, log_writer, context, 4)
	printParmsTxt(cnnOut, log_writer)

	log_writer.WriteString(fmt.Sprintln("==================================================================================="))
	log_writer.Flush()
	return cnnOut
}

// 목표 : time: 4m37.839912277s 보다는 빠르게!
func compactGappedConvolution(cnn TensorCipher, co, st, fh, fw int, convWgt, bnVar, bnWgt []float64, epsilon float64, context *Context) TensorCipher {
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
	evals := context.evals_
	threadNum := context.threadNum_

	mo = (co + ko*ko - 1) / (ko * ko)
	ctxt := make([][][]*rlwe.Ciphertext, mi)
	for m := 0; m < mi; m++ {
		ctxt[m] = make([][]*rlwe.Ciphertext, fh)
		for h := range fh {
			ctxt[m][h] = make([]*rlwe.Ciphertext, fw)
		}
	}
	checkTime := time.Now()
	var wg sync.WaitGroup
	wg.Add(threadNum)
	for thidx := range threadNum {
		go func() {
			defer wg.Done()
			for m := 0; m < mi; m++ {
				for h := 0; h < fh; h++ {
					for w := 0; w < fw; w++ {
						idx := m*fh*fw + h*fw + w
						if (thidx-idx)%threadNum != 0 {
							continue
						}
						r := ki*ki*wi*(h-(fh-1)/2) + ki*(w-(fw-1)/2)
						//ctxt[m][h][w], _ = evals[m].RotateNew(cnn.ciphers_[m], r)
						ctxt[m][h][w] = rotCtNew(cnn.ciphers_[m], r, context, thidx)
					}
				}
			}
		}()
	}
	wg.Wait()
	elapse := time.Since(checkTime)
	fmt.Println("ct rot end : ", elapse)

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
	selectOneVec := make([][]float64, co)
	for i := 0; i < co; i++ {
		bnValue := bnWgt[i] / math.Sqrt(bnVar[i]+epsilon)
		selectOneVec[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			i4 := j % (ko * wo)
			i3 := (j / (ko * wo)) % (ko * ho)
			if (ko*(i3%ko)+(i4%ko) == (i % (ko * ko))) && (j < ko*ko*wo*ho) {
				selectOneVec[i][j] = bnValue
			} else {
				selectOneVec[i][j] = 0
			}
		}
	}
	fmt.Println(selectOneVec[16][0])

	ctzero := ctZero(context, 0)
	initScale := cnn.ciphers_[0].Scale

	result_cipher := make([]*rlwe.Ciphertext, mo)
	for i := 0; i < mo; i++ {
		result_cipher[i] = ctzero.CopyNew()
	}

	checkTime = time.Now()
	wg.Add(threadNum)
	for thidx := range threadNum {
		go func() {
			defer wg.Done()
			for i3 := 0; i3 < co; i3++ {
				if (i3-thidx)%threadNum != 0 {
					continue
				}
				eval := evals[thidx]

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
					ctc := sumSlot(ctb, ki, 1, context, thidx)
					ctc = sumSlot(ctc, ki, ki*wi, context, thidx)
					eval.Add(cta, ctc, cta)
				}
				r := -int((i3%(ko*ko))/ko)*ko*wo - (i3 % ko)
				//temp, _ := eval.RotateNew(cta, r)
				temp := rotCtNew(cta, r, context, thidx)
				eval.Mul(temp, selectOneVec[i3], temp)
				eval.Add(result_cipher[int(i3/(ko*ko))], temp, result_cipher[int(i3/(ko*ko))])
			}
		}()
	}
	wg.Wait()
	elapse = time.Since(checkTime)
	fmt.Println("multyplexing end : ", elapse)

	for i := 0; i < mo; i++ {
		evals[0].RescaleTo(result_cipher[i], initScale, result_cipher[i])
	}
	result := NewTensorCipher(ko, ho, wo, co, mo, logn, result_cipher)
	return result
}

func compactGappedBatchNormPrint(cnn TensorCipher, bnbias, bnmean, bnvar, bnwgt []float64, epsilon, B float64, context *Context, stage int, log_writer *bufio.Writer) TensorCipher {
	timeStart := time.Now()
	cnnOut := compactGappedBatchNorm(cnn, bnbias, bnmean, bnvar, bnwgt, epsilon, B, context)
	elapse := time.Since(timeStart)
	str := fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("time: %s \n", elapse)
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("batchnorm %d-stage result\n", stage)
	fmt.Print(str)
	log_writer.WriteString(str)
	decryptPrintTxt(cnnOut.ciphers_, log_writer, context, 4)
	printParmsTxt(cnnOut, log_writer)

	str = fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)
	log_writer.Flush()
	return cnnOut
}

func compactGappedBatchNorm(cnn TensorCipher, bias, runningMean, runningVar, weight []float64, epsilon, B float64, context *Context) TensorCipher {
	h, w, c, k, m := cnn.h_, cnn.w_, cnn.c_, cnn.k_, cnn.m_
	n := 1 << cnn.logn_
	eval := context.evals_[0]

	cm_iVec := make([][]float64, m)
	for i := 0; i < m; i++ {
		cm_iVec[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			i4 := j % (k * w)
			i3 := (j / (k * w)) % (k * h)

			if j < k*k*w*h {
				index := k*(i3%k) + (i4 % k) + i*k*k
				cm_iVec[i][j] = bias[index] - weight[index]*runningMean[index]/math.Sqrt(runningVar[index]+epsilon)
				cm_iVec[i][j] /= B
			} else {
				cm_iVec[i][j] = 0.0
			}
		}
	}
	fmt.Printf("%f \n", cm_iVec[4][0])
	result_cipher := make([]*rlwe.Ciphertext, m)
	for i := 0; i < m; i++ {
		result_cipher[i], _ = eval.AddNew(cnn.ciphers_[i], cm_iVec[i])
	}

	cnnOut := NewTensorCipher(k, h, w, c, m, cnn.logn_, result_cipher)
	return cnnOut
}

func approxReLUPrint(cnn TensorCipher, alpha int, log_writer *bufio.Writer, context *Context, stage int) TensorCipher {
	timeStart := time.Now()
	cnnOut := approxReLU(cnn, alpha, context)
	elapse := time.Since(timeStart)
	str := fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("time: %s \n", elapse)
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("ReLU %d-stage result\n", stage)
	fmt.Print(str)
	log_writer.WriteString(str)
	decryptPrintTxt(cnnOut.ciphers_, log_writer, context, 4)
	printParmsTxt(cnnOut, log_writer)

	str = fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)
	log_writer.Flush()
	return cnnOut
}

func approxReLU(cnnIn TensorCipher, alpha int, context *Context) (cnnOut TensorCipher) {

	// parameter setting
	h, w, k, c, m, logn := cnnIn.h_, cnnIn.w_, cnnIn.k_, cnnIn.c_, cnnIn.m_, cnnIn.logn_
	ctxtIns := cnnIn.ciphers_
	threadNum := context.threadNum_
	result_cipher := make([]*rlwe.Ciphertext, m)

	var wg sync.WaitGroup
	wg.Add(threadNum)
	for threadidx := range threadNum {
		go func() {
			defer wg.Done()
			for i := 0; i < m; i++ {
				if (i-threadidx)%threadNum != 0 {
					continue
				}
				result_cipher[i] = EvalApproxMinimaxReLU(ctxtIns[i].CopyNew(), alpha, context, threadidx)
			}
		}()
	}
	wg.Wait()

	// temp = EvalApproxReLU(ctxtIn, alpha, context.eval_, context.params_)
	// temp = EvalApproxReLUDebug(ctxtIn, alpha, context, context.params_)

	cnnOut = NewTensorCipher(k, h, w, c, m, logn, result_cipher)

	return cnnOut

}

func maxPoolingPrint(cnn TensorCipher, st, fh, fw, alpha int, context *Context, stage int, log_writer *bufio.Writer) TensorCipher {
	timeStart := time.Now()
	cnnOut := maxPooling(cnn, st, fh, fw, alpha, context)
	elapse := time.Since(timeStart)

	str := fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("time: %s \n", elapse)
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("MaxPooling %d-stage result\n", stage)
	fmt.Print(str)
	log_writer.WriteString(str)
	decryptPrintTxt(cnnOut.ciphers_, log_writer, context, 4)
	printParmsTxt(cnnOut, log_writer)

	str = fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)
	log_writer.Flush()
	return cnn
}

func maxPooling(cnn TensorCipher, st, fh, fw, alpha int, context *Context) TensorCipher {
	if st != 2 || fh != 3 || fw != 3 {
		fmt.Println("error")
		os.Exit(0)
	}

	ki, hi, wi, ci, mi, logn := cnn.k_, cnn.h_, cnn.w_, cnn.c_, cnn.m_, cnn.logn_
	ko, ho, wo, co, mo := ki, hi, wi, ci, mi
	if st == 2 {
		ko, ho, wo, mo = ki*2, hi/2, wi/2, mi/(2*2)
	}
	n := 1 << logn
	threadNum := context.threadNum_
	if ko != 4 {
		fmt.Println("error")
		os.Exit(0)
	}

	masking := make([][][]float64, fh)
	for h := range fh {
		masking[h] = make([][]float64, fw)
		for w := range fw {
			masking[h][w] = make([]float64, n)
			wg := ki * (w - (fw-1)/2)
			hg := ki * (h - (fh-1)/2)
			for j := range n {
				i1 := j % (wi * ki)
				i2 := (j / (wi * ki)) % (hi * ki)
				w_ := (i1 + wg)
				h_ := (i2 + hg)
				if w_ >= 0 && w_ < ki*wi && h_ >= 0 && h_ < ki*hi && (j < ki*ki*hi*wi) {
					masking[h][w][j] = 1
				} else {
					masking[h][w][j] = 0
				}
			}
		}
	}
	maskingFinale := make([][]float64, 4)
	for i := 0; i < 4; i++ {
		maskingFinale[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			i4 := j % (ko * wo)
			i3 := (j / (ko * wo)) % (ko * ho)
			i1 := i % 2
			i2 := (i / 2) % 2
			if 2*i1 <= (i4%ko) && (i4%ko) < 2*(i1+1) && 2*i2 <= (i3%ko) && (i3%ko) < 2*(i2+1) && (j < ko*ko*wo*ho) {
				maskingFinale[i][j] = 1
			} else {
				maskingFinale[i][j] = 0
			}
		}
	}

	result_cipher := make([]*rlwe.Ciphertext, mo)
	for i := range mo {
		result_cipher[i] = ctZero(context, 0)
	}

	var wg sync.WaitGroup
	wg.Add(threadNum)
	for thidx := range threadNum {
		go func() {
			defer wg.Done()
			eval := context.evals_[thidx]
			btp := context.btp15s_[thidx]
			for m := 0; m < mi; m++ {
				if (m-thidx)%threadNum != 0 {
					continue
				}

				ctxt := make([][]*rlwe.Ciphertext, fh)
				for h := 0; h < fh; h++ {
					ctxt[h] = make([]*rlwe.Ciphertext, fw)
					for w := 0; w < fw; w++ {
						r := ki*ki*wi*(h-(fh-1)/2) + ki*(w-(fw-1)/2)
						ctxt[h][w] = rotCtNew(cnn.ciphers_[m], r, context, thidx)
						eval.Mul(ctxt[h][w], masking[h][w], ctxt[h][w])
						eval.Rescale(ctxt[h][w], ctxt[h][w])
						if m == 1 {
							fmt.Println("1 of print ", h, ", ", w)
							DecryptPrint(*context.params_, ctxt[h][w], *context.decryptor_, *context.encoders_[thidx])

						}
					}
				}

				fmt.Printf("%dth m is running... \n", m)
				var max_temp [3]*rlwe.Ciphertext
				for i := range 3 {
					max_temp[i] = EvalApproxMinimaxMax3(ctxt[i][0], ctxt[i][1], ctxt[i][2], alpha, context, thidx)
					max_temp[i], _ = btp.Bootstrap(max_temp[i])
					temp, _ := eval.ConjugateNew(max_temp[i])
					eval.Add(max_temp[i], temp, max_temp[i])
				}
				max := EvalApproxMinimaxMax3(max_temp[0], max_temp[1], max_temp[2], alpha, context, thidx)
				r := ki*(m%2) + ki*ki*wi*((m/2)%2)
				maxRot := rotCtNew(max, -r, context, thidx)
				eval.Mul(maxRot, maskingFinale[m%4], maxRot)
				fmt.Println(m, "th is end")
				DecryptPrint(*context.params_, maxRot, *context.decryptor_, *context.encoders_[thidx])
				eval.Add(result_cipher[m/4], maxRot, result_cipher[m/4])

			}
		}()
	}
	wg.Wait()

	wg.Add(threadNum)
	for thidx := range threadNum {
		eval := context.evals_[thidx]
		go func() {
			defer wg.Done()
			for i := range mo {
				if (i-thidx)%threadNum != 0 {
					continue
				}
				eval.Rescale(result_cipher[i], result_cipher[i])
			}
		}()
	}
	wg.Wait()

	cnnOut := NewTensorCipher(ko, ho, wo, co, mo, logn, result_cipher)
	return cnnOut
}

func bootstrapPrint(cnn TensorCipher, context *Context, stage int, log_writer *bufio.Writer) TensorCipher {
	timeStart := time.Now()
	cnnOut := bootstrapImageImaginary(cnn, context)
	elapse := time.Since(timeStart)

	str := fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("time: %s \n", elapse)
	fmt.Print(str)
	log_writer.WriteString(str)

	str = fmt.Sprintf("Bootstrapp %d-stage result\n", stage)
	fmt.Print(str)
	log_writer.WriteString(str)
	decryptPrintTxt(cnnOut.ciphers_, log_writer, context, 4)
	printParmsTxt(cnnOut, log_writer)

	str = fmt.Sprintln("===================================================================================")
	fmt.Print(str)
	log_writer.WriteString(str)
	log_writer.Flush()
	return cnnOut
}
func bootstrapImageImaginary(cnnIn TensorCipher, context *Context) (cnnOut TensorCipher) {
	// parameters
	mi, ki, hi, wi, ci, logn := cnnIn.m_, cnnIn.k_, cnnIn.h_, cnnIn.w_, cnnIn.c_, cnnIn.logn_
	mo, ko, ho, wo, co := mi, ki, hi, wi, ci

	result_cipher := make([]*rlwe.Ciphertext, mo)
	threadNum := context.threadNum_

	var wg sync.WaitGroup
	wg.Add(threadNum)
	for threadidx := range threadNum {
		go func() {
			defer wg.Done()

			btp := context.btp15s_[threadidx]
			eval := context.evals_[threadidx]
			for i := 0; i < mo; i++ {
				if (i-threadidx)%threadNum != 0 {
					continue
				}
				fmt.Printf("boot %d: \n", i)
				var err error
				result_cipher[i], err = btp.Bootstrap(cnnIn.ciphers_[i])
				if err != nil {
					fmt.Println(err)
				}
				temp2, _ := eval.ConjugateNew(result_cipher[i]) // imaginary removing bootstrapping
				eval.Add(result_cipher[i], temp2, result_cipher[i])
			}
		}()
	}
	wg.Wait()
	cnnOut = NewTensorCipher(ko, ho, wo, co, mo, logn, result_cipher)
	return cnnOut
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
		context.encoders_[0].Encode(data[i], plaintext)
		cipher, err := context.encryptors_[0].EncryptNew(plaintext)
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
	encoders_   []*hefloat.Encoder
	encryptors_ []*rlwe.Encryptor
	decryptor_  *rlwe.Decryptor
	sk_         *rlwe.SecretKey
	pk_         *rlwe.PublicKey
	rts_        *map[int]struct{}
	btp15s_     []*bootstrapping.Evaluator
	evals_      []*hefloat.Evaluator
	params_     *hefloat.Parameters
	threadNum_  int
}

func NewContext(encoder []*hefloat.Encoder, encryptor []*rlwe.Encryptor, decryptor *rlwe.Decryptor, sk *rlwe.SecretKey,
	pk *rlwe.PublicKey, rts *map[int]struct{}, btp15 []*bootstrapping.Evaluator, eval []*hefloat.Evaluator, params *hefloat.Parameters, threadNum int) *Context {
	result := Context{
		encoders_:   encoder,
		encryptors_: encryptor,
		decryptor_:  decryptor,
		sk_:         sk,
		pk_:         pk,
		rts_:        rts,
		btp15s_:     btp15,
		evals_:      eval,
		params_:     params,
		threadNum_:  threadNum,
	}
	return &result
}
func decryptPrint(ciphertexts []*rlwe.Ciphertext, context *Context, num int) {
	params := *context.params_
	decryptor := *context.decryptor_
	encoder := *context.encoders_[0]
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
	encoder := *context.encoders_[0]
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
