package main

import (
	"github.com/lifejade/imag-merge-resnet/cnn"
)

func main() {
	cnn.ResNetImageNetMultyThread(18, 0, 64)
}
