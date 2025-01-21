package cnn

import (
	"fmt"
	"math"
	"math/big"
	"runtime"
	"strconv"
	"sync"
	"time"

	"bufio"
	"log"
	"os"

	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he/hefloat"
	"github.com/tuneinsight/lattigo/v5/he/hefloat/bootstrapping"
)

func ResNetImageNetMultyThread(layerNum int, imageID, threadNum int) {
	//CPU full power
	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

	//check layernumber
	if !(layerNum == 18 || layerNum == 32) {
		fmt.Println("layer_num is not correct")
		os.Exit(1)
	}
	// load inference image
	var file *os.File
	file, err := os.Open("testFiles/test_values.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	SkipLines(in, 224*224*3*imageID)

	image := make([][]float64, 3)
	for i := 0; i < 3; i++ {
		image[i] = ReadLines(in, image[i], 224*224)
	}
	fmt.Println("image load end")

	//ckks parameter init
	ckksParams := CNN_Cifar18_Parameters
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
	var context *Context

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
	encoders := make([]*hefloat.Encoder, threadNum)
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

			} else {
				btps[i] = btp.ShallowCopy()
				// btps[i], _ = bootstrapping.NewEvaluator(btpParams15,btpevk)
			}
			//generate -er
			encoders[i] = hefloat.NewEncoder(params)
			encryptors[i] = rlwe.NewEncryptor(params, pk)
			evaluators[i] = hefloat.NewEvaluator(params, evk)
		}()
	}
	wg.Wait()
	context = NewContext(encoders, encryptors, decryptor, sk, pk, &rts, btps, evaluators, &params, threadNum)
	fmt.Println("context init end")

	// result image label
	allTimeStart := time.Now()
	var buffer []int
	file, err = os.Open("testFiles/test_label.txt")
	if err != nil {
		log.Fatal(err)
	}
	in = bufio.NewScanner(file)
	SkipLines(in, imageID)
	buffer = ReadLinesInt(in, buffer, 1)
	imageLabel := buffer[0]
	// output files
	file, _ = os.Create("log/resnet" + strconv.Itoa(layerNum) + "_ImageNet_image" + strconv.Itoa(imageID) + ".txt")
	log_writer := bufio.NewWriter(file)
	//result file init
	file, _ = os.Create("result/resnet" + strconv.Itoa(layerNum) + "_ImageNet_label" + strconv.Itoa(imageID) + ".txt")
	result_writer := bufio.NewWriter(file)
	defer file.Close()

	label, maxScore := ResNetImageNet(layerNum, image, context, log_writer)
	fmt.Println("image label: ", imageLabel)
	fmt.Println("inferred label: ", label)
	fmt.Println("max score: ", maxScore)
	log_writer.WriteString("image label: " + strconv.Itoa(imageLabel) + "\n")
	log_writer.WriteString("inferred label: " + strconv.Itoa(label) + "\n")
	log_writer.WriteString("max score: " + fmt.Sprintf("%f", maxScore) + "\n")
	log_writer.Flush()
	result_writer.WriteString("image_id: " + strconv.Itoa(imageID) + ", image label: " + strconv.Itoa(imageLabel) + ", inferred label: " + strconv.Itoa(label) + "\n")
	result_writer.Flush()
	allElapse := time.Since(allTimeStart)
	fmt.Printf("all threads time : %s \n", allElapse)
}

func ResNetImageNet(layerNum int, image [][]float64, context *Context, log_writer *bufio.Writer) (inferredLabel int, maxScore float64) {
	fmt.Println("Start Resnet with ImageNet!!!!!!")
	evaluators := context.evals_
	params := context.params_
	encoder, decryptor := context.encoders_[0], context.decryptor_
	co, st, fh, fw := 64, 2, 7, 7
	logp := 46
	logn := 16
	n := 1 << logn
	stage := 0
	epsilon := 0.00001
	B := 40.0
	linWgt, linBias, convBNWgt, shortcutWgt, blockNum := ImportParametersImageNet(layerNum)

	for i := 0; i < 3; i++ {
		for len(image[i]) < n {
			image[i] = append(image[i], 0.0)
		}
		for j := 0; j < n; j++ {
			image[i][j] /= B
		}
	}

	cnn := NewTensorCipherFormData(1, 224, 224, 3, 3, logn, logp, image, context)
	fmt.Println("preprocess & encrypt inference image end")
	fmt.Println(context.params_.MaxLevel())

	//start level = 33(19 + 14)
	fmt.Println("drop level")
	fmt.Printf("start level : %d, currenlevel : %d\n", cnn.ciphers_[0].Level(), cnn.ciphers_[0].Level()-31)
	for i := 0; i < cnn.m_; i++ {
		evaluators[i].DropLevel(cnn.ciphers_[i], 31)
	}

	totalTimeStart := time.Now()
	_ = totalTimeStart
	_, _, _ = params, encoder, decryptor
	cnn = compactGappedConvolutionPrint(cnn, co, st, fh, fw, convBNWgt[stage].convwgt, convBNWgt[stage].bnvar, convBNWgt[stage].bnwgt, epsilon, context, stage, log_writer)
	cnn = compactGappedBatchNormPrint(cnn, convBNWgt[stage].bnbias, convBNWgt[stage].bnmean, convBNWgt[stage].bnvar, convBNWgt[stage].bnwgt, epsilon, B, context, stage, log_writer)
	for i := 0; i < cnn.m_; i++ {
		evaluators[i].SetScale(cnn.ciphers_[i], rlwe.NewScale(1<<logp))
	}
	cnn = bootstrapPrint(cnn, context, stage, log_writer)
	cnn = approxReLUPrint(cnn, 14, log_writer, context, stage)
	cnn = bootstrapPrint(cnn, context, stage, log_writer)

	cnn = maxPoolingPrint(cnn, 2, 3, 3, 14, context, stage, log_writer)
	os.Exit(0)
	stage++

	for largeBlockID := 0; largeBlockID < 4; largeBlockID++ {
		if largeBlockID == 0 {
			co = 64
		} else if largeBlockID == 1 {
			co = 128
		} else if largeBlockID == 2 {
			co = 256
		} else if largeBlockID == 3 {
			co = 512
		}
		endNum := blockNum[largeBlockID]

		for blockID := 0; blockID <= endNum; blockID++ {
			// stage = 2*((endNum+1)*largeBlockID+blockID) + 1
			fmt.Println("layer ", stage)
			tempCnn := CopyTensorCipher(cnn)
			if largeBlockID >= 1 && blockID == 0 {
				st = 2
			} else {
				st = 1
			}
			cnn = compactGappedConvolutionPrint(cnn, co, st, fh, fw, convBNWgt[stage].convwgt, convBNWgt[stage].bnvar, convBNWgt[stage].bnwgt, epsilon, context, stage, log_writer)
			cnn = compactGappedBatchNormPrint(cnn, convBNWgt[stage].bnbias, convBNWgt[stage].bnmean, convBNWgt[stage].bnvar, convBNWgt[stage].bnwgt, epsilon, B, context, stage, log_writer)
			cnn = bootstrapPrint(cnn, context, stage, log_writer)
			cnn = approxReLUPrint(cnn, 13, log_writer, context, stage)

			stage++
			fmt.Println("layer ", stage)
			st = 1

			cnn = compactGappedConvolutionPrint(cnn, co, st, fh, fw, convBNWgt[stage].convwgt, convBNWgt[stage].bnvar, convBNWgt[stage].bnwgt, epsilon, context, stage, log_writer)
			cnn = compactGappedBatchNormPrint(cnn, convBNWgt[stage].bnbias, convBNWgt[stage].bnmean, convBNWgt[stage].bnvar, convBNWgt[stage].bnwgt, epsilon, B, context, stage, log_writer)
			if largeBlockID >= 1 && blockID == 0 {
				idx := largeBlockID - 1
				tempCnn = compactGappedConvolutionPrint(tempCnn, co, 1, 1, 1, shortcutWgt[idx].convwgt, shortcutWgt[idx].bnvar, shortcutWgt[idx].bnwgt, epsilon, context, stage, log_writer)
				tempCnn = compactGappedBatchNormPrint(tempCnn, shortcutWgt[idx].bnbias, shortcutWgt[idx].bnmean, shortcutWgt[idx].bnvar, shortcutWgt[idx].bnwgt, epsilon, B, context, stage, log_writer)
			}
			cnn = cipherAddPrint(cnn, tempCnn, context, stage, log_writer)
			cnn = bootstrapPrint(cnn, context, stage, log_writer)
			cnn = approxReLUPrint(cnn, 13, log_writer, context, stage)
			stage++
		}
	}
	cnn = averagepoolingPrint(cnn, B, context, log_writer)
	cnn = fullyConnectedPrint(cnn, linWgt, linBias, 1000, 512, context, log_writer)

	elapse := time.Since(totalTimeStart)
	fmt.Printf("Done in %s \n", elapse)

	if cnn.m_ != 1 || len(cnn.ciphers_) != 1 {
		fmt.Println("error, m is not 1")
		os.Exit(1)
	}

	decryptPrintTxt(cnn.ciphers_, log_writer, context, 10)

	rtnVec := make([]complex128, params.LogMaxSlots())
	encoder.Decode(decryptor.DecryptNew(cnn.ciphers_[0]), rtnVec)
	fmt.Printf("(")
	log_writer.WriteString("(")
	for i := 0; i < 9; i++ {
		fmt.Print(rtnVec[i], ", ")
		log_writer.WriteString(fmt.Sprintf("%6.10f", rtnVec[i]) + ", ")
	}
	fmt.Print(rtnVec[9], ")\n")
	log_writer.WriteString(fmt.Sprintf("%6.10f", rtnVec[9]) + ")\n")
	fmt.Println("total time: ", elapse)
	log_writer.WriteString("total time: " + elapse.String() + "\n")

	inferredLabel = 0
	maxScore = -100.0
	for i := 0; i < 10; i++ {
		if maxScore < real(rtnVec[i]) {
			inferredLabel = i
			maxScore = real(rtnVec[i])
		}
	}
	return inferredLabel, maxScore
}

func ImportParametersImageNet(layerNum int) (linwgt []float64, linbias []float64, convblockwgt []*WeightConvBN, shortcutwgt []*WeightConvBN, blockNum []int) {

	var dir string

	// directory name
	if layerNum != 18 && layerNum != 32 {
		fmt.Println("layer number is not valid")
	}
	if layerNum == 18 {
		dir = "resnet18_new"
	} else if layerNum == 34 {
		dir = "resnet34_new"
	} else {
		dir = ""
	}

	// endNum
	blockNum = make([]int, 4)
	if layerNum == 18 {
		copy(blockNum, []int{2, 2, 2, 2})
	} else if layerNum == 34 {
		copy(blockNum, []int{3, 4, 6, 3})
	} else {
		fmt.Println("layer_num is not correct")
	}

	var num_c, num_b, num_m, num_v, num_w int
	// var num_c int

	convblockwgt = make([]*WeightConvBN, layerNum-1)
	shortcutwgt = make([]*WeightConvBN, 3)

	for i := 0; i < layerNum-1; i++ {
		var temp WeightConvBN
		convblockwgt[i] = &temp
	}
	for i := 0; i < 3; i++ {
		var temp WeightConvBN
		shortcutwgt[i] = &temp
	}

	// convolution parameters
	fh, fw, ci, co := 7, 7, 3, 64
	convblockwgt[0].convwgt = make([]float64, fh*fw*ci*co)
	ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/conv1_weight.txt", &convblockwgt[0].convwgt, fh*fw*ci*co)
	num_c++
	// convolution parameters
	for j := 1; j <= 4; j++ {
		for k := 0; k < blockNum[j-1]; k++ {
			// co setting
			if j == 1 {
				co = 64
			} else if j == 2 {
				co = 128
			} else if j == 3 {
				co = 256
			} else if j == 4 {
				co = 512
			}

			// ci setting
			if j == 1 || (j == 2 && k == 0) {
				ci = 64
			} else if (j == 2 && k != 0) || (j == 3 && k == 0) {
				ci = 128
			} else if (j == 3 && k != 0) || (j == 4 && k == 0) {
				ci = 256
			} else {
				ci = 512
			}
			convblockwgt[num_c].convwgt = make([]float64, fh*fw*ci*co)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_conv1_weight.txt", &convblockwgt[num_c].convwgt, fh*fw*ci*co)
			num_c++

			// ci setting
			if j == 1 {
				ci = 64
			} else if j == 2 {
				ci = 128
			} else if j == 3 {
				ci = 256
			} else {
				ci = 512
			}
			convblockwgt[num_c].convwgt = make([]float64, fh*fw*ci*co)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_conv2_weight.txt", &convblockwgt[num_c].convwgt, fh*fw*ci*co)
			num_c++
		}
	}
	fmt.Println("conv parameters load end")

	// batch_normalization parameters
	ci = 64
	convblockwgt[0].bnbias = make([]float64, ci)
	ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/bn1_bias.txt", &convblockwgt[0].bnbias, ci)
	num_b++
	convblockwgt[0].bnmean = make([]float64, ci)
	ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/bn1_running_mean.txt", &convblockwgt[0].bnmean, ci)
	num_m++
	convblockwgt[0].bnvar = make([]float64, ci)
	ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/bn1_running_var.txt", &convblockwgt[0].bnvar, ci)
	num_v++
	convblockwgt[0].bnwgt = make([]float64, ci)
	ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/bn1_weight.txt", &convblockwgt[0].bnwgt, ci)
	num_w++

	// batch_normalization parameters
	for j := 1; j <= 4; j++ {
		if j == 1 {
			ci = 64
		} else if j == 2 {
			ci = 128
		} else if j == 3 {
			ci = 256
		} else if j == 4 {
			ci = 512
		}

		for k := 0; k < blockNum[j-1]; k++ {
			convblockwgt[num_b].bnbias = make([]float64, ci)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_bias.txt", &convblockwgt[num_b].bnbias, ci)
			num_b++

			convblockwgt[num_m].bnmean = make([]float64, ci)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_running_mean.txt", &convblockwgt[num_m].bnmean, ci)
			num_m++

			convblockwgt[num_v].bnvar = make([]float64, ci)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_running_var.txt", &convblockwgt[num_v].bnvar, ci)
			num_v++

			convblockwgt[num_w].bnwgt = make([]float64, ci)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn1_weight.txt", &convblockwgt[num_w].bnwgt, ci)
			num_w++

			convblockwgt[num_b].bnbias = make([]float64, ci)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_bias.txt", &convblockwgt[num_b].bnbias, ci)
			num_b++

			convblockwgt[num_m].bnmean = make([]float64, ci)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_running_mean.txt", &convblockwgt[num_m].bnmean, ci)
			num_m++

			convblockwgt[num_v].bnvar = make([]float64, ci)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_running_var.txt", &convblockwgt[num_v].bnvar, ci)
			num_v++

			convblockwgt[num_w].bnwgt = make([]float64, ci)
			ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+"_bn2_weight.txt", &convblockwgt[num_w].bnwgt, ci)
			num_w++
		}
	}
	fmt.Println("batchnorm parameters load end")

	for i := 0; i < 3; i++ {
		// shortcut convolution parameters
		ci := (i + 1) * 64
		co := (i + 2) * 64
		shortcutwgt[i].convwgt = make([]float64, ci*co)
		ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(i+2)+"_0_downsample_0_weight.txt", &shortcutwgt[i].convwgt, ci*co)

		// shortcut batch normalization parameters
		ci = (i + 1) * 128
		shortcutwgt[i].bnbias = make([]float64, ci)
		ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(i+2)+"_0_downsample_1_bias.txt", &shortcutwgt[i].bnbias, ci)
		shortcutwgt[i].bnmean = make([]float64, ci)
		ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(i+2)+"_0_downsample_1_running_mean.txt", &shortcutwgt[i].bnmean, ci)
		shortcutwgt[i].bnvar = make([]float64, ci)
		ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(i+2)+"_0_downsample_1_running_var.txt", &shortcutwgt[i].bnvar, ci)
		shortcutwgt[i].bnwgt = make([]float64, ci)
		ReadConvWgtIdx("parameters/resnet_pretrained/"+dir+"/layer"+strconv.Itoa(i+2)+"_0_downsample_1_weight.txt", &shortcutwgt[i].bnwgt, ci)
	}
	fmt.Println("shortcut parameters load end")

	// FC layer
	file, err := os.Open("parameters/resnet_pretrained/" + dir + "/fc_weight.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in := bufio.NewScanner(file)
	linwgt = make([]float64, 1000*512)
	linwgt = ReadLines(in, linwgt, 1000*512)
	file, err = os.Open("parameters/resnet_pretrained/" + dir + "/fc_bias.txt")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	in = bufio.NewScanner(file)
	linbias = make([]float64, 1000)
	linbias = ReadLines(in, linbias, 1000)
	fmt.Println("fc parameters load end")

	return linwgt, linbias, convblockwgt, shortcutwgt, blockNum
}

func DecryptPrint(params hefloat.Parameters, ciphertext *rlwe.Ciphertext, decryptor rlwe.Decryptor, encoder hefloat.Encoder) {

	n := 1 << ciphertext.LogSlots()
	message := make([]complex128, n)
	encoder.Decode(decryptor.DecryptNew(ciphertext), message)

	fmt.Println()
	fmt.Printf("Level: %d (logQ = %d)\n", ciphertext.Level(), params.LogQLvl(ciphertext.Level()))
	fmt.Printf("Scale: 2^%f\n", ciphertext.LogScale())
	fmt.Printf("Values: %6.10f %6.10f %6.10f %6.10f %6.10f...\n", message[0], message[1], message[2], message[3], message[4])

	// max, min
	max, min := 0.0, 1.0
	for _, v := range message {
		if max < real(v) {
			max = real(v)
		}
		if min > real(v) {
			min = real(v)
		}
	}

	fmt.Println("Max, Min value: ", max, " ", min)
	fmt.Println()
}

func ReadConvWgtIdx(fileName string, storeVal *[]float64, lineNum int) {
	in, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer in.Close()

	scanner := bufio.NewScanner(in)
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := strconv.ParseFloat(s, 64)
		// fmt.Println(p)
		(*storeVal)[i] = p
	}
}

func SkipLines(scanner *bufio.Scanner, lineNum int) {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		scanner.Text()
	}
}

func ReadLines(scanner *bufio.Scanner, storeVal []float64, lineNum int) []float64 {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := strconv.ParseFloat(s, 64)
		storeVal = append(storeVal, p)
	}

	return storeVal
}

func ReadLinesInt(scanner *bufio.Scanner, storeVal []int, lineNum int) []int {
	for i := 0; i < lineNum; i++ {
		scanner.Scan()
		s := scanner.Text()
		p, _ := strconv.Atoi(s)
		storeVal = append(storeVal, p)
	}

	return storeVal
}

// func ResNetImageNetMultipleImage(layerNum int, startImageID, endImageID int) {
// 	//CPU full power
// 	runtime.GOMAXPROCS(runtime.NumCPU()) // CPU 개수를 구한 뒤 사용할 최대 CPU 개수 설정
// 	fmt.Println("Maximum number of CPUs: ", runtime.GOMAXPROCS(0))

// 	//check layernumber
// 	if !(layerNum == 18 || layerNum == 32) {
// 		fmt.Println("layer_num is not correct")
// 		os.Exit(1)
// 	}
// 	// load inference image
// 	var file *os.File
// 	file, err := os.Open("testFiles/test_values.txt")
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	defer file.Close()
// 	in := bufio.NewScanner(file)
// 	SkipLines(in, 224*224*3*startImageID)

// 	threadNum := endImageID - startImageID + 1
// 	images := make([][][]float64, threadNum)
// 	for i := 0; i < threadNum; i++ {
// 		images[i] = make([][]float64, 3)
// 		for j := 0; j < 3; j++ {
// 			images[i][j] = ReadLines(in, images[i][j], 224*224)
// 		}
// 	}
// 	fmt.Println("image load end")

// 	//ckks parameter init
// 	ckksParams := CNN_Cifar18_Parameters
// 	initparams, err := hefloat.NewParametersFromLiteral(ckksParams.SchemeParams)
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Println("ckks parameter init end")

// 	bootParams15 := ckksParams.BootstrappingParams
// 	//bootParams14.LogSlots, bootParams13.LogSlots, bootParams12.LogSlots = utils.Pointy(14), utils.Pointy(13), utils.Pointy(12)
// 	btpParams15, err := bootstrapping.NewParametersFromLiteral(initparams, bootParams15)
// 	if err != nil {
// 		panic(err)
// 	}
// 	btpParams15.SlotsToCoeffsParameters.Scaling = new(big.Float).SetFloat64(0.5)
// 	//bootParams14 := ckksParams.BootstrappingParams
// 	//btpParams14, err := bootstrapping.NewParametersFromLiteral(params,bootParams14)
// 	fmt.Println("bootstrapping parameter init end")

// 	// generate keys
// 	//fmt.Println("generate keys")
// 	//keytime := time.Now()
// 	initkgen := rlwe.NewKeyGenerator(initparams)
// 	initsk := initkgen.GenSecretKeyNew()
// 	contexts := make([]*Context, threadNum)

// 	var params hefloat.Parameters
// 	var pk *rlwe.PublicKey
// 	var sk *rlwe.SecretKey
// 	var rlk *rlwe.RelinearizationKey
// 	var rtk []*rlwe.GaloisKey

// 	parThreadNum := 4
// 	var wg_th sync.WaitGroup
// 	wg_th.Add(parThreadNum)
// 	for i := 0; i < threadNum; i++ {
// 		//generate bootstrapper
// 		btparr := make([]*bootstrapping.Evaluator, parThreadNum)
// 		var btpevk15_ *bootstrapping.EvaluationKeys
// 		btpevk15_, sk, _ = btpParams15.GenEvaluationKeys(initsk)
// 		for j := 0; j < parThreadNum; j++ {
// 			go func() {
// 				defer wg_th.Done()
// 				btparr[j], _ = bootstrapping.NewEvaluator(btpParams15, btpevk15_)
// 			}()
// 		}
// 		wg_th.Wait()
// 		runtime.GC()
// 		btp15 := btparr[0]

// 		fmt.Println("generated bootstrapper end")
// 		if i == 0 {
// 			params = *btp15.GetParameters()
// 		}
// 		kgen := rlwe.NewKeyGenerator(params)
// 		if i == 0 {
// 			pk = kgen.GenPublicKeyNew(sk)
// 			rlk = kgen.GenRelinearizationKeyNew(sk)
// 			// generate keys - Rotating key
// 			convRot := []int{0, 1, 2, 3, 4, 8, 16, 32, 221, 222, 223, 224, 225, 226, 227, 445, 446, 447, 448, 449, 450, 451, 669, 670, 671, 672, 673, 674, 675, 892, 896, 900, 1784, 1792, 1800, 3568, 3584, 3600, 7136, 7168, 7200, 58336, 58368, 58400, 61936, 61952, 61968, 62145, 62146, 62147, 62148, 62149, 62150, 62151, 62152, 62153, 62154, 62155, 62156, 62157, 62158, 62159, 62160, 62161, 62162, 62163, 62164, 62165, 62166, 62167, 62168, 62169, 62170, 62171, 62172, 62173, 62174, 62175, 62176, 62369, 62370, 62371, 62372, 62373, 62374, 62375, 62376, 62377, 62378, 62379, 62380, 62381, 62382, 62383, 62384, 62385, 62386, 62387, 62388, 62389, 62390, 62391, 62392, 62393, 62394, 62395, 62396, 62397, 62398, 62399, 62400, 62593, 62594, 62595, 62596, 62597, 62598, 62599, 62600, 62601, 62602, 62603, 62604, 62605, 62606, 62607, 62608, 62609, 62610, 62611, 62612, 62613, 62614, 62615, 62616, 62617, 62618, 62619, 62620, 62621, 62622, 62623, 62624, 62817, 62818, 62819, 62820, 62821, 62822, 62823, 62824, 62825, 62826, 62827, 62828, 62829, 62830, 62831, 62832, 62833, 62834, 62835, 62836, 62837, 62838, 62839, 62840, 62841, 62842, 62843, 62844, 62845, 62846, 62847, 62848, 63041, 63042, 63043, 63044, 63045, 63046, 63047, 63048, 63049, 63050, 63051, 63052, 63053, 63054, 63055, 63056, 63057, 63058, 63059, 63060, 63061, 63062, 63063, 63064, 63065, 63066, 63067, 63068, 63069, 63070, 63071, 63072, 63265, 63266, 63267, 63268, 63269, 63270, 63271, 63272, 63273, 63274, 63275, 63276, 63277, 63278, 63279, 63280, 63281, 63282, 63283, 63284, 63285, 63286, 63287, 63288, 63289, 63290, 63291, 63292, 63293, 63294, 63295, 63296, 63489, 63490, 63491, 63492, 63493, 63494, 63495, 63496, 63497, 63498, 63499, 63500, 63501, 63502, 63503, 63504, 63505, 63506, 63507, 63508, 63509, 63510, 63511, 63512, 63513, 63514, 63515, 63516, 63517, 63518, 63519, 63520, 63713, 63714, 63715, 63716, 63717, 63718, 63719, 63720, 63721, 63722, 63723, 63724, 63725, 63726, 63727, 63728, 63729, 63730, 63731, 63732, 63733, 63734, 63735, 63736, 63737, 63738, 63739, 63740, 63741, 63742, 63743, 63744, 63752, 63937, 63938, 63939, 63940, 63941, 63942, 63943, 63944, 63945, 63946, 63947, 63948, 63949, 63950, 63951, 63952, 63953, 63954, 63955, 63956, 63957, 63958, 63959, 63960, 63961, 63962, 63963, 63964, 63965, 63966, 63967, 63968, 64161, 64162, 64163, 64164, 64165, 64166, 64167, 64168, 64169, 64170, 64171, 64172, 64173, 64174, 64175, 64176, 64177, 64178, 64179, 64180, 64181, 64182, 64183, 64184, 64185, 64186, 64187, 64188, 64189, 64190, 64191, 64192, 64385, 64386, 64387, 64388, 64389, 64390, 64391, 64392, 64393, 64394, 64395, 64396, 64397, 64398, 64399, 64400, 64401, 64402, 64403, 64404, 64405, 64406, 64407, 64408, 64409, 64410, 64411, 64412, 64413, 64414, 64415, 64416, 64609, 64610, 64611, 64612, 64613, 64614, 64615, 64616, 64617, 64618, 64619, 64620, 64621, 64622, 64623, 64624, 64625, 64626, 64627, 64628, 64629, 64630, 64631, 64632, 64633, 64634, 64635, 64636, 64637, 64638, 64639, 64640, 64644, 64833, 64834, 64835, 64836, 64837, 64838, 64839, 64840, 64841, 64842, 64843, 64844, 64845, 64846, 64847, 64848, 64849, 64850, 64851, 64852, 64853, 64854, 64855, 64856, 64857, 64858, 64859, 64860, 64861, 64862, 64863, 64864, 64865, 64866, 64867, 65057, 65058, 65059, 65060, 65061, 65062, 65063, 65064, 65065, 65066, 65067, 65068, 65069, 65070, 65071, 65072, 65073, 65074, 65075, 65076, 65077, 65078, 65079, 65080, 65081, 65082, 65083, 65084, 65085, 65086, 65087, 65088, 65089, 65090, 65091, 65281, 65282, 65283, 65284, 65285, 65286, 65287, 65288, 65289, 65290, 65291, 65292, 65293, 65294, 65295, 65296, 65297, 65298, 65299, 65300, 65301, 65302, 65303, 65304, 65305, 65306, 65307, 65308, 65309, 65310, 65311, 65312, 65313, 65314, 65315, 65504, 65505, 65506, 65507, 65508, 65509, 65510, 65511, 65512, 65513, 65514, 65515, 65516, 65517, 65518, 65519, 65520, 65521, 65522, 65523, 65524, 65525, 65526, 65527, 65528, 65529, 65530, 65531, 65532, 65533, 65534, 65535}
// 			galEls := make([]uint64, len(convRot))
// 			for i, x := range convRot {
// 				galEls[i] = params.GaloisElement(x)
// 			}
// 			galEls = append(galEls, params.GaloisElementForComplexConjugation())

// 			rtk = make([]*rlwe.GaloisKey, len(galEls))
// 			var wg sync.WaitGroup
// 			wg.Add(len(galEls))
// 			for i := range galEls {
// 				i := i

// 				go func() {
// 					defer wg.Done()
// 					kgen_ := rlwe.NewKeyGenerator(params)
// 					rtk[i] = kgen_.GenGaloisKeyNew(galEls[i], sk)
// 				}()
// 			}
// 			wg.Wait()
// 		}

// 		evk := rlwe.NewMemEvaluationKeySet(rlk, rtk...)
// 		//generate -er
// 		encryptor := rlwe.NewEncryptor(params, pk)
// 		decryptor := rlwe.NewDecryptor(params, sk)
// 		encoder := hefloat.NewEncoder(params)
// 		evaluator := hefloat.NewEvaluator(params, evk)
// 		fmt.Println("generate Evaluator end")

// 		contexts[i] = NewContext(encoder, encryptor, decryptor, sk, pk, btparr, rtk, rlk, evaluator, &params)
// 	}

// 	// result image label
// 	allTimeStart := time.Now()
// 	var wg sync.WaitGroup
// 	wg.Add(threadNum)
// 	for i := 0; i < threadNum; i++ {
// 		image := images[i]
// 		imageID := startImageID + i
// 		context := contexts[i]

// 		go func() {
// 			defer wg.Done()
// 			var buffer []int
// 			var file *os.File
// 			file, err = os.Open("testFiles/test_label.txt")
// 			if err != nil {
// 				log.Fatal(err)
// 			}
// 			in = bufio.NewScanner(file)
// 			SkipLines(in, imageID)
// 			buffer = ReadLinesInt(in, buffer, 1)
// 			imageLabel := buffer[0]
// 			// output files
// 			file, _ = os.Create("log/resnet" + strconv.Itoa(layerNum) + "_ImageNet_image" + strconv.Itoa(imageID) + ".txt")
// 			log_writer := bufio.NewWriter(file)
// 			//result file init
// 			file, _ = os.Create("result/resnet" + strconv.Itoa(layerNum) + "_ImageNet_label" + strconv.Itoa(imageID) + ".txt")
// 			result_writer := bufio.NewWriter(file)
// 			defer file.Close()

// 			label, maxScore := ResNetImageNet(layerNum, image, context, log_writer)
// 			fmt.Println("image label: ", imageLabel)
// 			fmt.Println("inferred label: ", label)
// 			fmt.Println("max score: ", maxScore)
// 			log_writer.WriteString("image label: " + strconv.Itoa(imageLabel) + "\n")
// 			log_writer.WriteString("inferred label: " + strconv.Itoa(label) + "\n")
// 			log_writer.WriteString("max score: " + fmt.Sprintf("%f", maxScore) + "\n")
// 			log_writer.Flush()
// 			result_writer.WriteString("image_id: " + strconv.Itoa(imageID) + ", image label: " + strconv.Itoa(imageLabel) + ", inferred label: " + strconv.Itoa(label) + "\n")
// 			result_writer.Flush()
// 		}()
// 	}
// 	wg.Wait()
// 	allElapse := time.Since(allTimeStart)
// 	fmt.Printf("all threads time : %s \n", allElapse)
// }
