package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDatasetInspect(t *testing.T) {
	// Create a temp file with known content
	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	content := "Hello world\nThis is a test\nThird line\n"
	os.WriteFile(path, []byte(content), 0644)

	// Just verify it doesn't panic — output goes to stdout
	datasetInspect(path)
}

func TestFormatBytes(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{500, "500 B"},
		{1500, "1.5 KB"},
		{1500000, "1.5 MB"},
		{1500000000, "1.5 GB"},
	}
	for _, tt := range tests {
		got := formatBytes(tt.n)
		if got != tt.want {
			t.Errorf("formatBytes(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}

func TestFormatCount(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{500, "500"},
		{1500, "1.5K"},
		{1500000, "1.5M"},
		{1500000000, "1.5B"},
	}
	for _, tt := range tests {
		got := formatCount(tt.n)
		if got != tt.want {
			t.Errorf("formatCount(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}
