//go:build !darwin || !cgo

package main

import "log"

func cmdTrainMetal() {
	log.Fatal("Metal training requires macOS with CGO_ENABLED=1")
}
