package main

import (
	"testing"
)

func TestParseKV(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want map[string]string
	}{
		{
			name: "empty",
			args: []string{},
			want: map[string]string{},
		},
		{
			name: "single key=value",
			args: []string{"data=train.txt"},
			want: map[string]string{"data": "train.txt"},
		},
		{
			name: "multiple key=value",
			args: []string{"model=TinyLlama", "data=train.txt", "steps=1000"},
			want: map[string]string{"model": "TinyLlama", "data": "train.txt", "steps": "1000"},
		},
		{
			name: "positional args",
			args: []string{"qwen2", "hello world"},
			want: map[string]string{"_0": "qwen2", "_1": "hello world"},
		},
		{
			name: "mixed positional and kv",
			args: []string{"data=train.txt", "mymodel", "steps=500"},
			want: map[string]string{"data": "train.txt", "_0": "mymodel", "steps": "500"},
		},
		{
			name: "value with equals sign",
			args: []string{"lr=1e-5"},
			want: map[string]string{"lr": "1e-5"},
		},
		{
			name: "path values",
			args: []string{"data=/home/user/data/train.txt", "model=/opt/models/llama"},
			want: map[string]string{"data": "/home/user/data/train.txt", "model": "/opt/models/llama"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseKV(tt.args)
			if len(got) != len(tt.want) {
				t.Errorf("parseKV() returned %d entries, want %d\n  got:  %v\n  want: %v", len(got), len(tt.want), got, tt.want)
				return
			}
			for k, wantV := range tt.want {
				if gotV, ok := got[k]; !ok {
					t.Errorf("parseKV() missing key %q", k)
				} else if gotV != wantV {
					t.Errorf("parseKV()[%q] = %q, want %q", k, gotV, wantV)
				}
			}
		})
	}
}

func TestKVInt(t *testing.T) {
	args := map[string]string{"steps": "5000", "dim": "512"}

	if got := kvInt(args, "steps", 100); got != 5000 {
		t.Errorf("kvInt(steps) = %d, want 5000", got)
	}
	if got := kvInt(args, "dim", 128); got != 512 {
		t.Errorf("kvInt(dim) = %d, want 512", got)
	}
	if got := kvInt(args, "layers", 4); got != 4 {
		t.Errorf("kvInt(layers) = %d, want 4 (default)", got)
	}
}

func TestKVFloat(t *testing.T) {
	args := map[string]string{"lr": "1e-5", "wd": "0.1"}

	if got := kvFloat(args, "lr", 6e-4); got != 1e-5 {
		t.Errorf("kvFloat(lr) = %v, want 1e-5", got)
	}
	if got := kvFloat(args, "wd", 0.01); got != 0.1 {
		t.Errorf("kvFloat(wd) = %v, want 0.1", got)
	}
	if got := kvFloat(args, "beta1", 0.9); got != 0.9 {
		t.Errorf("kvFloat(beta1) = %v, want 0.9 (default)", got)
	}
}

func TestDetectBestBackend(t *testing.T) {
	backend := detectBestBackend()
	// Just verify it returns a valid string
	switch backend {
	case "cuda-kernels", "graph", "cpu":
		// ok
	default:
		t.Errorf("detectBestBackend() = %q, want one of cuda-kernels/graph/cpu", backend)
	}
}

func TestResolveModelPath(t *testing.T) {
	// Test that absolute paths pass through
	if got := resolveModelPath("/tmp"); got != "/tmp" {
		t.Errorf("resolveModelPath(/tmp) = %q, want /tmp", got)
	}
}

func TestFormatParams(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{500, "500"},
		{1500, "1.5K"},
		{623700, "623.7K"},
		{1100000, "1.1M"},
		{7000000000, "7.00B"},
	}
	for _, tt := range tests {
		got := formatParams(tt.n)
		if got != tt.want {
			t.Errorf("formatParams(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}
