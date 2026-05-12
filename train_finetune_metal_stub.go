//go:build !darwin || !cgo

package main

func cmdFinetuneMetalLoRA(modelPath, dataPath string, steps int, lr float64, rank int, logEvery int) {
}
