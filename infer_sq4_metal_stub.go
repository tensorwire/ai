//go:build !darwin || !cgo

package main

import "github.com/tensorwire/mongoose"

func cmdInferSQ4Metal(path, prompt string, metal *mongoose.Metal, te mongoose.TensorEngine,
	cfg map[string]interface{},
	dim, nLayers, heads, kvHeads, ffnDim, vocabSize, headDim, kvDim, attnDim, maxSeq int,
	ropeTheta, normEps float32) {
}
