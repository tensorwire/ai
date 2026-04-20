package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// findLatestCheckpoint finds the most recent checkpoint directory.
func findLatestCheckpoint(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			dirs = append(dirs, e.Name())
		}
	}
	if len(dirs) == 0 {
		return ""
	}
	sort.Strings(dirs)
	return filepath.Join(dir, dirs[len(dirs)-1])
}

// listCheckpoints returns sorted checkpoint directories.
func listCheckpoints(dir string) []string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil
	}
	var dirs []string
	for _, e := range entries {
		if e.IsDir() && !strings.HasPrefix(e.Name(), ".") {
			dirs = append(dirs, filepath.Join(dir, e.Name()))
		}
	}
	sort.Strings(dirs)
	return dirs
}

// ckptMeta is the metadata saved alongside a checkpoint.
type ckptMeta struct {
	Step  int     `json:"step"`
	Loss  float64 `json:"loss"`
	LR    float64 `json:"lr"`
	Epoch int     `json:"epoch,omitempty"`
}

// readCkptMeta reads meta.json from a checkpoint directory.
func readCkptMeta(dir string) (*ckptMeta, error) {
	data, err := os.ReadFile(filepath.Join(dir, "meta.json"))
	if err != nil {
		return nil, err
	}
	var m ckptMeta
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// writeCkptMeta writes meta.json to a checkpoint directory.
func writeCkptMeta(dir string, m *ckptMeta) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "meta.json"), data, 0644)
}

// cmdCheckpoint handles checkpoint subcommands.
//
//	ai checkpoint ls [dir]
//	ai checkpoint diff <ckpt1> <ckpt2>
func cmdCheckpoint(args map[string]string) {
	sub := args["_0"]

	switch sub {
	case "ls", "list":
		cmdCheckpointLs(args)
	case "diff":
		cmdCheckpointDiff(args)
	default:
		fmt.Fprintln(os.Stderr, "Usage:")
		fmt.Fprintln(os.Stderr, "  ai checkpoint ls [dir]              List checkpoints")
		fmt.Fprintln(os.Stderr, "  ai checkpoint diff <ckpt1> <ckpt2>  Compare two checkpoints")
		os.Exit(1)
	}
}

func cmdCheckpointLs(args map[string]string) {
	dir := args["_1"]
	if dir == "" {
		home, _ := os.UserHomeDir()
		dir = filepath.Join(home, ".ai", "checkpoints")
	}

	ckpts := listCheckpoints(dir)
	if len(ckpts) == 0 {
		fmt.Printf("No checkpoints found in %s\n", dir)
		return
	}

	fmt.Printf("Checkpoints in %s:\n\n", dir)
	fmt.Printf("  %-30s %8s %10s %10s %10s\n", "Name", "Step", "Loss", "LR", "Size")
	fmt.Printf("  %-30s %8s %10s %10s %10s\n", "----", "----", "----", "--", "----")

	for _, ckpt := range ckpts {
		name := filepath.Base(ckpt)
		meta, err := readCkptMeta(ckpt)

		stepStr := "-"
		lossStr := "-"
		lrStr := "-"
		if err == nil {
			stepStr = fmt.Sprintf("%d", meta.Step)
			lossStr = fmt.Sprintf("%.4f", meta.Loss)
			lrStr = fmt.Sprintf("%.1e", meta.LR)
		}

		sizeStr := dirSize(ckpt)

		fmt.Printf("  %-30s %8s %10s %10s %10s\n", name, stepStr, lossStr, lrStr, sizeStr)
	}

	latest := findLatestCheckpoint(dir)
	if latest != "" {
		fmt.Printf("\n  latest: %s\n", filepath.Base(latest))
	}
}

func cmdCheckpointDiff(args map[string]string) {
	ckpt1 := args["_1"]
	ckpt2 := args["_2"]
	if ckpt1 == "" || ckpt2 == "" {
		fmt.Fprintln(os.Stderr, "Usage: ai checkpoint diff <ckpt1> <ckpt2>")
		os.Exit(1)
	}

	m1, err1 := readCkptMeta(ckpt1)
	m2, err2 := readCkptMeta(ckpt2)

	fmt.Printf("Checkpoint diff:\n")
	fmt.Printf("  A: %s\n", filepath.Base(ckpt1))
	fmt.Printf("  B: %s\n", filepath.Base(ckpt2))
	fmt.Println()

	if err1 != nil || err2 != nil {
		if err1 != nil {
			fmt.Printf("  A: no meta.json (%v)\n", err1)
		}
		if err2 != nil {
			fmt.Printf("  B: no meta.json (%v)\n", err2)
		}
		return
	}

	fmt.Printf("  %-12s %12s %12s %12s\n", "", "A", "B", "Delta")
	fmt.Printf("  %-12s %12s %12s %12s\n", "----", "----", "----", "-----")
	fmt.Printf("  %-12s %12d %12d %+12d\n", "step", m1.Step, m2.Step, m2.Step-m1.Step)
	fmt.Printf("  %-12s %12.4f %12.4f %+12.4f\n", "loss", m1.Loss, m2.Loss, m2.Loss-m1.Loss)
	fmt.Printf("  %-12s %12.1e %12.1e\n", "lr", m1.LR, m2.LR)

	s1 := dirSizeBytes(ckpt1)
	s2 := dirSizeBytes(ckpt2)
	fmt.Printf("  %-12s %12s %12s\n", "size", formatBytes(s1), formatBytes(s2))
}

// dirSize returns the total size of files in a directory as a formatted string.
func dirSize(dir string) string {
	return formatBytes(dirSizeBytes(dir))
}

// dirSizeBytes returns the total size of files in a directory.
func dirSizeBytes(dir string) int {
	var total int64
	filepath.Walk(dir, func(_ string, info os.FileInfo, _ error) error {
		if info != nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return int(total)
}
