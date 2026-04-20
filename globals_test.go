package main

import (
	"testing"
)

func TestSelectBackend(t *testing.T) {
	tests := []struct {
		device string
		want   string
	}{
		{"auto", "auto"},
		{"cuda", "cuda"},
		{"metal", "metal"},
		{"cpu", "cpu"},
		{"webgpu", "webgpu"},
		{"cuda:0", "cuda"},
		{"cuda:1", "cuda"},
	}
	for _, tt := range tests {
		GlobalDevice = tt.device
		got := SelectBackend()
		if got != tt.want {
			t.Errorf("SelectBackend(%q) = %q, want %q", tt.device, got, tt.want)
		}
	}
	GlobalDevice = "auto" // reset
}

func TestInt32Float32Roundtrip(t *testing.T) {
	for _, v := range []int32{0, 1, -1, 42, 255, 32000, -32000} {
		f := int32ToFloat32Bits(v)
		back := float32ToInt32Bits(f)
		if back != v {
			t.Errorf("roundtrip(%d): got %d", v, back)
		}
	}
}
