package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFindLatestCheckpoint(t *testing.T) {
	dir := t.TempDir()

	if got := findLatestCheckpoint(dir); got != "" {
		t.Errorf("empty dir: got %q, want empty", got)
	}

	os.Mkdir(filepath.Join(dir, "step-00100"), 0755)
	os.Mkdir(filepath.Join(dir, "step-00200"), 0755)
	os.Mkdir(filepath.Join(dir, "step-00050"), 0755)

	got := findLatestCheckpoint(dir)
	want := filepath.Join(dir, "step-00200")
	if got != want {
		t.Errorf("findLatestCheckpoint() = %q, want %q", got, want)
	}
}

func TestListCheckpoints(t *testing.T) {
	dir := t.TempDir()

	if got := listCheckpoints(dir); len(got) != 0 {
		t.Errorf("empty: got %d checkpoints, want 0", len(got))
	}

	os.Mkdir(filepath.Join(dir, "ckpt-1"), 0755)
	os.Mkdir(filepath.Join(dir, "ckpt-2"), 0755)
	os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("hi"), 0644)
	os.Mkdir(filepath.Join(dir, ".hidden"), 0755)

	got := listCheckpoints(dir)
	if len(got) != 2 {
		t.Errorf("got %d checkpoints, want 2: %v", len(got), got)
	}
}

func TestFindLatestCheckpointNonexistent(t *testing.T) {
	got := findLatestCheckpoint("/nonexistent/path")
	if got != "" {
		t.Errorf("nonexistent dir: got %q, want empty", got)
	}
}

func TestCkptMetaRoundtrip(t *testing.T) {
	dir := t.TempDir()

	m := &ckptMeta{Step: 5000, Loss: 2.34, LR: 1e-4}
	if err := writeCkptMeta(dir, m); err != nil {
		t.Fatalf("write: %v", err)
	}

	got, err := readCkptMeta(dir)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if got.Step != 5000 {
		t.Errorf("Step = %d, want 5000", got.Step)
	}
	if got.Loss != 2.34 {
		t.Errorf("Loss = %f, want 2.34", got.Loss)
	}
	if got.LR != 1e-4 {
		t.Errorf("LR = %f, want 1e-4", got.LR)
	}
}

func TestReadCkptMetaMissing(t *testing.T) {
	_, err := readCkptMeta(t.TempDir())
	if err == nil {
		t.Error("expected error for missing meta.json")
	}
}

func TestDirSizeBytes(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.bin"), make([]byte, 1000), 0644)
	os.WriteFile(filepath.Join(dir, "b.bin"), make([]byte, 2000), 0644)

	got := dirSizeBytes(dir)
	if got != 3000 {
		t.Errorf("dirSizeBytes = %d, want 3000", got)
	}
}

func TestCheckpointLsWithMeta(t *testing.T) {
	dir := t.TempDir()

	ckpt1 := filepath.Join(dir, "step-01000")
	os.Mkdir(ckpt1, 0755)
	writeCkptMeta(ckpt1, &ckptMeta{Step: 1000, Loss: 3.5, LR: 6e-4})
	os.WriteFile(filepath.Join(ckpt1, "weights.bin"), make([]byte, 500), 0644)

	ckpt2 := filepath.Join(dir, "step-02000")
	os.Mkdir(ckpt2, 0755)
	writeCkptMeta(ckpt2, &ckptMeta{Step: 2000, Loss: 2.1, LR: 3e-4})
	os.WriteFile(filepath.Join(ckpt2, "weights.bin"), make([]byte, 500), 0644)

	// Just verify it doesn't panic
	args := map[string]string{"_0": "ls", "_1": dir}
	cmdCheckpointLs(args)
}

func TestCheckpointDiffWithMeta(t *testing.T) {
	dir := t.TempDir()

	ckpt1 := filepath.Join(dir, "step-01000")
	os.Mkdir(ckpt1, 0755)
	writeCkptMeta(ckpt1, &ckptMeta{Step: 1000, Loss: 3.5, LR: 6e-4})

	ckpt2 := filepath.Join(dir, "step-05000")
	os.Mkdir(ckpt2, 0755)
	writeCkptMeta(ckpt2, &ckptMeta{Step: 5000, Loss: 1.8, LR: 1e-4})

	args := map[string]string{"_0": "diff", "_1": ckpt1, "_2": ckpt2}
	cmdCheckpointDiff(args)
}
