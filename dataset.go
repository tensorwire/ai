package main

import (
	"fmt"
	"log"
	"os"
	"sort"
)

// cmdDataset handles dataset subcommands.
//
//	ai dataset inspect <file>
func cmdDataset(args map[string]string) {
	sub := args["_0"]
	file := args["_1"]
	if file == "" {
		file = args["file"]
	}

	switch sub {
	case "inspect":
		if file == "" {
			fmt.Fprintln(os.Stderr, "Usage: ai dataset inspect <file>")
			os.Exit(1)
		}
		datasetInspect(file)
	default:
		fmt.Fprintln(os.Stderr, "Usage: ai dataset inspect <file>")
		os.Exit(1)
	}
}

// datasetInspect analyzes a text dataset and reports statistics.
func datasetInspect(path string) {
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatalf("read: %v", err)
	}

	totalBytes := len(raw)
	totalLines := 0
	totalWords := 0
	lineLengths := make([]int, 0)
	byteFreq := make(map[byte]int)
	inWord := false
	lineLen := 0

	for _, b := range raw {
		byteFreq[b]++
		if b == '\n' {
			totalLines++
			lineLengths = append(lineLengths, lineLen)
			lineLen = 0
			inWord = false
		} else {
			lineLen++
			if b == ' ' || b == '\t' {
				inWord = false
			} else if !inWord {
				inWord = true
				totalWords++
			}
		}
	}
	if lineLen > 0 {
		totalLines++
		lineLengths = append(lineLengths, lineLen)
	}

	sort.Ints(lineLengths)
	var avgLine float64
	if len(lineLengths) > 0 {
		total := 0
		for _, l := range lineLengths {
			total += l
		}
		avgLine = float64(total) / float64(len(lineLengths))
	}

	uniqueBytes := len(byteFreq)

	// Estimate token count (rough: ~4 chars per token for English)
	estTokens := totalBytes / 4

	fmt.Printf("ai dataset inspect — %s\n\n", path)
	fmt.Printf("  size:        %s\n", formatBytes(totalBytes))
	fmt.Printf("  lines:       %d\n", totalLines)
	fmt.Printf("  words:       %d\n", totalWords)
	fmt.Printf("  unique bytes: %d / 256\n", uniqueBytes)
	fmt.Printf("  est. tokens: ~%s (byte-level: %s)\n", formatCount(estTokens), formatCount(totalBytes))
	fmt.Println()

	if len(lineLengths) > 0 {
		p50 := lineLengths[len(lineLengths)/2]
		p95 := lineLengths[int(float64(len(lineLengths))*0.95)]
		p99 := lineLengths[int(float64(len(lineLengths))*0.99)]
		fmt.Printf("  line length:\n")
		fmt.Printf("    min:  %d\n", lineLengths[0])
		fmt.Printf("    avg:  %.0f\n", avgLine)
		fmt.Printf("    p50:  %d\n", p50)
		fmt.Printf("    p95:  %d\n", p95)
		fmt.Printf("    p99:  %d\n", p99)
		fmt.Printf("    max:  %d\n", lineLengths[len(lineLengths)-1])
	}
	fmt.Println()

	// Recommended training config
	fmt.Printf("  Recommended:\n")
	if totalBytes < 1_000_000 {
		fmt.Printf("    seq_len:  64\n")
		fmt.Printf("    steps:    %d\n", totalBytes/64*10)
	} else if totalBytes < 100_000_000 {
		fmt.Printf("    seq_len:  128\n")
		fmt.Printf("    steps:    %d\n", min(totalBytes/128*3, 50000))
	} else {
		fmt.Printf("    seq_len:  256\n")
		fmt.Printf("    steps:    50000\n")
	}
}

func formatBytes(b int) string {
	switch {
	case b >= 1_000_000_000:
		return fmt.Sprintf("%.1f GB", float64(b)/1e9)
	case b >= 1_000_000:
		return fmt.Sprintf("%.1f MB", float64(b)/1e6)
	case b >= 1_000:
		return fmt.Sprintf("%.1f KB", float64(b)/1e3)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func formatCount(n int) string {
	switch {
	case n >= 1_000_000_000:
		return fmt.Sprintf("%.1fB", float64(n)/1e9)
	case n >= 1_000_000:
		return fmt.Sprintf("%.1fM", float64(n)/1e6)
	case n >= 1_000:
		return fmt.Sprintf("%.1fK", float64(n)/1e3)
	default:
		return fmt.Sprintf("%d", n)
	}
}
